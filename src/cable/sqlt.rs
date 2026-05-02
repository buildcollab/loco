//! SQLite-backed pub/sub.
//!
//! Persists every published payload as a row in `loco_cable_messages` and
//! polls the table for new rows. Each subscriber maintains its own last-seen
//! id (held in memory) so deliveries fan out without the messages being
//! consumed.
//!
//! For multi-node deployments prefer [`super::redis`]; SQLite is local-disk
//! and only useful for single-node "no Redis, no Postgres" setups, but it
//! does survive a process restart.

use std::{
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use sqlx::{sqlite::SqlitePoolOptions, Row, SqlitePool};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    cable::pubsub::{PubSub, Subscription},
    config::SqliteCableConfig,
    Result,
};

const ENSURE_TABLE_SQL: &str = r"
CREATE TABLE IF NOT EXISTS loco_cable_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
";

const ENSURE_INDEX_SQL: &str = r"
CREATE INDEX IF NOT EXISTS idx_loco_cable_messages_topic_id
    ON loco_cable_messages (topic, id);
";

#[derive(Default)]
struct Topics {
    /// `topic -> list of (last_seen_id, sender)`
    subs: DashMap<String, Vec<TopicSub>>,
}

struct TopicSub {
    id: u64,
    last_seen: Arc<AtomicI64>,
    tx: mpsc::UnboundedSender<Bytes>,
}

pub struct SqlitePubSub {
    pool: SqlitePool,
    topics: Arc<Topics>,
    polling_interval: Duration,
    retention_minutes: u32,
    next_sub_id: AtomicI64,
    shutdown: CancellationToken,
}

impl SqlitePubSub {
    /// Connect, ensure schema, and spawn the polling task.
    pub async fn connect(cfg: &SqliteCableConfig) -> Result<Arc<Self>> {
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect(&cfg.uri)
            .await
            ?;
        sqlx::query(ENSURE_TABLE_SQL)
            .execute(&pool)
            .await
            ?;
        sqlx::query(ENSURE_INDEX_SQL)
            .execute(&pool)
            .await
            ?;

        if cfg.dangerously_flush {
            tracing::warn!("cable_sqlt: flush mode enabled, truncating loco_cable_messages");
            sqlx::query("DELETE FROM loco_cable_messages")
                .execute(&pool)
                .await
                ?;
        }

        let me = Arc::new(Self {
            pool,
            topics: Arc::new(Topics::default()),
            polling_interval: Duration::from_millis(cfg.polling_interval_ms),
            retention_minutes: cfg.retention_minutes,
            next_sub_id: AtomicI64::new(1),
            shutdown: CancellationToken::new(),
        });

        let bg = me.clone();
        tokio::spawn(async move { bg.run_polling().await });
        Ok(me)
    }

    async fn run_polling(self: Arc<Self>) {
        let mut interval = tokio::time::interval(self.polling_interval);
        let mut last_gc = std::time::Instant::now();
        loop {
            tokio::select! {
                () = self.shutdown.cancelled() => break,
                _ = interval.tick() => {}
            }
            self.poll_once().await;
            if last_gc.elapsed() > Duration::from_secs(60) {
                self.gc().await;
                last_gc = std::time::Instant::now();
            }
        }
    }

    async fn poll_once(&self) {
        // Snapshot subscriber state, then query per topic.
        type SubSnapshot = (u64, Arc<AtomicI64>, mpsc::UnboundedSender<Bytes>);
        let topics_snapshot: Vec<(String, Vec<SubSnapshot>)> = self
            .topics
            .subs
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    entry
                        .value()
                        .iter()
                        .map(|s| (s.id, s.last_seen.clone(), s.tx.clone()))
                        .collect(),
                )
            })
            .collect();

        for (topic, subs) in topics_snapshot {
            // Find the minimum last_seen across subs to pull only what's needed.
            let Some(min_last) = subs.iter().map(|(_, l, _)| l.load(Ordering::Acquire)).min()
            else {
                continue;
            };

            let rows = match sqlx::query(
                "SELECT id, payload FROM loco_cable_messages \
                 WHERE topic = ?1 AND id > ?2 ORDER BY id ASC LIMIT 500",
            )
            .bind(&topic)
            .bind(min_last)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows,
                Err(err) => {
                    tracing::warn!(error = %err, topic, "cable_sqlt: poll failed");
                    continue;
                }
            };

            for row in rows {
                let id: i64 = row.get(0);
                let payload: Vec<u8> = row.get(1);
                let payload = Bytes::from(payload);
                for (_, last_seen, tx) in &subs {
                    if last_seen.load(Ordering::Acquire) >= id {
                        continue;
                    }
                    let _ = tx.send(payload.clone());
                    last_seen.store(id, Ordering::Release);
                }
            }
        }
    }

    async fn gc(&self) {
        let cutoff_minutes = self.retention_minutes;
        let _ = sqlx::query(
            "DELETE FROM loco_cable_messages WHERE created_at < datetime('now', ?1)",
        )
        .bind(format!("-{cutoff_minutes} minutes"))
        .execute(&self.pool)
        .await
        .map_err(|err| {
            tracing::debug!(error = %err, "cable_sqlt: GC failed");
            err
        });
    }
}

#[async_trait]
impl PubSub for SqlitePubSub {
    async fn publish(&self, topic: &str, payload: Bytes) -> Result<()> {
        sqlx::query("INSERT INTO loco_cable_messages (topic, payload) VALUES (?1, ?2)")
            .bind(topic)
            .bind(payload.as_ref())
            .execute(&self.pool)
            .await
            ?;
        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Subscription> {
        // Start from the current max id so the subscriber only sees new
        // messages (existing rows aren't replayed).
        let max_id: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(id), 0) FROM loco_cable_messages WHERE topic = ?1",
        )
        .bind(topic)
        .fetch_one(&self.pool)
        .await
        ?;

        let last_seen = Arc::new(AtomicI64::new(max_id));
        let (tx, rx) = mpsc::unbounded_channel();
        let sub_id = self.next_sub_id.fetch_add(1, Ordering::Relaxed) as u64;

        self.topics
            .subs
            .entry(topic.to_string())
            .or_default()
            .push(TopicSub {
                id: sub_id,
                last_seen,
                tx,
            });

        Ok(Subscription::new(
            rx,
            SqltDropGuard {
                topics: self.topics.clone(),
                topic: topic.to_string(),
                sub_id,
            },
        ))
    }
}

struct SqltDropGuard {
    topics: Arc<Topics>,
    topic: String,
    sub_id: u64,
}

impl Drop for SqltDropGuard {
    fn drop(&mut self) {
        if let Some(mut entry) = self.topics.subs.get_mut(&self.topic) {
            entry.retain(|s| s.id != self.sub_id);
            let empty = entry.is_empty();
            drop(entry);
            if empty {
                self.topics.subs.remove(&self.topic);
            }
        }
    }
}

/// Build a [`PubSub`] from configuration.
///
/// # Errors
/// Returns an error if the SQLite pool cannot be created or the schema
/// migration fails.
pub async fn create_provider(cfg: &SqliteCableConfig) -> Result<Arc<dyn PubSub>> {
    let provider: Arc<SqlitePubSub> = SqlitePubSub::connect(cfg).await?;
    Ok(provider as Arc<dyn PubSub>)
}
