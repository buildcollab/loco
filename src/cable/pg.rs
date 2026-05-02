//! Postgres-backed pub/sub.
//!
//! Persists every published payload as a row in `loco_cable_messages` and
//! polls the table for new rows. Each subscriber maintains its own last-seen
//! id so deliveries fan out without messages being consumed.
//!
//! Multi-node deployments work as long as every Loco process points at the
//! same Postgres URL. For higher-throughput / lower-latency fan-out use
//! [`super::redis`].

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
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    cable::pubsub::{PubSub, Subscription},
    config::PostgresCableConfig,
    Result,
};

const ENSURE_TABLE_SQL: &str = r"
CREATE TABLE IF NOT EXISTS loco_cable_messages (
    id BIGSERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    payload BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
";

const ENSURE_INDEX_SQL: &str = r"
CREATE INDEX IF NOT EXISTS idx_loco_cable_messages_topic_id
    ON loco_cable_messages (topic, id);
";

#[derive(Default)]
struct Topics {
    subs: DashMap<String, Vec<TopicSub>>,
}

struct TopicSub {
    id: u64,
    last_seen: Arc<AtomicI64>,
    tx: mpsc::UnboundedSender<Bytes>,
}

pub struct PgPubSub {
    pool: PgPool,
    topics: Arc<Topics>,
    polling_interval: Duration,
    retention_minutes: u32,
    next_sub_id: AtomicI64,
    shutdown: CancellationToken,
}

impl PgPubSub {
    pub async fn connect(cfg: &PostgresCableConfig) -> Result<Arc<Self>> {
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .connect(&cfg.uri)
            .await?;
        sqlx::query(ENSURE_TABLE_SQL).execute(&pool).await?;
        sqlx::query(ENSURE_INDEX_SQL).execute(&pool).await?;

        if cfg.dangerously_flush {
            tracing::warn!("cable_pg: flush mode enabled, truncating loco_cable_messages");
            sqlx::query("TRUNCATE loco_cable_messages")
                .execute(&pool)
                .await?;
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
            let Some(min_last) = subs.iter().map(|(_, l, _)| l.load(Ordering::Acquire)).min()
            else {
                continue;
            };

            let rows = match sqlx::query(
                "SELECT id, payload FROM loco_cable_messages \
                 WHERE topic = $1 AND id > $2 ORDER BY id ASC LIMIT 500",
            )
            .bind(&topic)
            .bind(min_last)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows,
                Err(err) => {
                    tracing::warn!(error = %err, topic, "cable_pg: poll failed");
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
        let cutoff_minutes = self.retention_minutes as i64;
        let _ = sqlx::query(
            "DELETE FROM loco_cable_messages WHERE created_at < NOW() - ($1::bigint || ' minutes')::interval",
        )
        .bind(cutoff_minutes)
        .execute(&self.pool)
        .await
        .map_err(|err| {
            tracing::debug!(error = %err, "cable_pg: GC failed");
            err
        });
    }
}

#[async_trait]
impl PubSub for PgPubSub {
    async fn publish(&self, topic: &str, payload: Bytes) -> Result<()> {
        sqlx::query("INSERT INTO loco_cable_messages (topic, payload) VALUES ($1, $2)")
            .bind(topic)
            .bind(payload.as_ref())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Subscription> {
        let max_id: Option<i64> = sqlx::query_scalar(
            "SELECT MAX(id) FROM loco_cable_messages WHERE topic = $1",
        )
        .bind(topic)
        .fetch_one(&self.pool)
        .await?;
        let max_id = max_id.unwrap_or(0);

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
            PgDropGuard {
                topics: self.topics.clone(),
                topic: topic.to_string(),
                sub_id,
            },
        ))
    }
}

struct PgDropGuard {
    topics: Arc<Topics>,
    topic: String,
    sub_id: u64,
}

impl Drop for PgDropGuard {
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
/// Returns an error if the Postgres pool cannot be created or the schema
/// migration fails.
pub async fn create_provider(cfg: &PostgresCableConfig) -> Result<Arc<dyn PubSub>> {
    let provider = PgPubSub::connect(cfg).await?;
    Ok(provider as Arc<dyn PubSub>)
}
