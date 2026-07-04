//! PostgreSQL Message Queue (`pgmq` extension) backend.
//!
//! pgmq is a *queue* (read-and-delete + visibility timeout), not pub/sub —
//! a single shared queue would deliver each message to one consumer rather
//! than fanning out. To get pub/sub semantics on top, this backend keeps
//! **one ephemeral pgmq queue per live subscription** plus a tiny
//! `loco_cable_pgmq_subs` table that maps `topic -> [queue_name, ...]`. On
//! `publish`, we insert the payload into every queue currently subscribed
//! to that topic (`pgmq.send`). On `subscribe`, we create a queue, register
//! it in the table, and start a polling loop that calls `pgmq.read` /
//! `pgmq.delete`. On drop we deregister and `pgmq.drop_queue`.
//!
//! Trade-off: durable / FIFO-per-subscriber / at-least-once, but with the
//! cost of a queue per connection. Fine for ~thousands of connections per
//! node; for higher fan-out use [`super::redis`].

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use pgmq::PGMQueueExt;
use serde_json::Value;
use sqlx::{postgres::PgPoolOptions, PgPool};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    cable::pubsub::{PubSub, Subscription},
    config::PgMQCableConfig,
    Error, Result,
};

const ENSURE_TABLE_SQL: &str = r"
CREATE TABLE IF NOT EXISTS loco_cable_pgmq_subs (
    queue_name TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_loco_cable_pgmq_subs_topic
    ON loco_cable_pgmq_subs (topic);
";

pub struct PgMqPubSub {
    pool: PgPool,
    queue: PGMQueueExt,
    visibility_timeout_sec: i32,
    batch_size: i32,
    polling_interval: Duration,
}

impl PgMqPubSub {
    pub async fn connect(cfg: &PgMQCableConfig) -> Result<Arc<Self>> {
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .connect(&cfg.uri)
            .await?;
        sqlx::query(ENSURE_TABLE_SQL).execute(&pool).await?;
        let queue = PGMQueueExt::new(cfg.uri.clone(), 8)
            .await
            .map_err(Error::wrap)?;
        // Ensure the pgmq extension is installed (no-op if already present).
        queue.init().await.map_err(Error::wrap)?;
        Ok(Arc::new(Self {
            pool,
            queue,
            visibility_timeout_sec: cfg.visibility_timeout_sec,
            batch_size: cfg.batch_size,
            polling_interval: Duration::from_millis(cfg.polling_interval_ms),
        }))
    }

    fn make_queue_name() -> String {
        // pgmq queue names are limited; ulid base32 is short and safe.
        format!(
            "loco_cable_{}",
            ulid::Ulid::new().to_string().to_lowercase()
        )
    }
}

#[async_trait]
impl PubSub for PgMqPubSub {
    async fn publish(&self, topic: &str, payload: Bytes) -> Result<()> {
        // Look up all live subscriber queues for this topic.
        let queue_names: Vec<String> =
            sqlx::query_scalar("SELECT queue_name FROM loco_cable_pgmq_subs WHERE topic = $1")
                .bind(topic)
                .fetch_all(&self.pool)
                .await?;

        // pgmq stores JSON. We base64-wrap raw bytes inside a JSON envelope so
        // both text and binary payloads round-trip cleanly.
        let payload_str = match std::str::from_utf8(&payload) {
            Ok(s) => Value::String(s.to_owned()),
            Err(_) => Value::String(format!("base64:{}", b64(&payload))),
        };
        let envelope = serde_json::json!({ "p": payload_str });
        for q in queue_names {
            if let Err(err) = self.queue.send(&q, &envelope).await {
                // Stale subscription rows happen when a process crashes
                // before it can clean up. Log and move on.
                tracing::debug!(error = %err, queue = q, "cable_pgmq: send failed (likely stale sub)");
            }
        }
        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Subscription> {
        let queue_name = Self::make_queue_name();
        self.queue.create(&queue_name).await.map_err(Error::wrap)?;
        sqlx::query("INSERT INTO loco_cable_pgmq_subs (queue_name, topic) VALUES ($1, $2)")
            .bind(&queue_name)
            .bind(topic)
            .execute(&self.pool)
            .await?;

        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<Bytes>();

        let task = PgMqReader {
            queue: self.queue.clone(),
            queue_name: queue_name.clone(),
            visibility_timeout_sec: self.visibility_timeout_sec,
            batch_size: self.batch_size,
            polling_interval: self.polling_interval,
            tx,
            cancel: cancel.clone(),
        };
        tokio::spawn(task.run());

        Ok(Subscription::new(
            rx,
            PgMqDropGuard {
                cancel,
                pool: self.pool.clone(),
                queue: self.queue.clone(),
                queue_name,
            },
        ))
    }
}

struct PgMqReader {
    queue: PGMQueueExt,
    queue_name: String,
    visibility_timeout_sec: i32,
    batch_size: i32,
    polling_interval: Duration,
    tx: mpsc::UnboundedSender<Bytes>,
    cancel: CancellationToken,
}

impl PgMqReader {
    async fn run(self) {
        let mut ticker = tokio::time::interval(self.polling_interval);
        loop {
            tokio::select! {
                () = self.cancel.cancelled() => break,
                _ = ticker.tick() => {}
            }
            let msgs = match self
                .queue
                .read_batch::<Value>(
                    &self.queue_name,
                    self.visibility_timeout_sec,
                    self.batch_size,
                )
                .await
            {
                Ok(m) => m,
                Err(err) => {
                    tracing::debug!(error = %err, queue = %self.queue_name, "cable_pgmq: read failed");
                    continue;
                }
            };
            if msgs.is_empty() {
                continue;
            }
            for m in msgs {
                let id = m.msg_id;
                let payload = extract_payload(&m.message);
                if self.tx.send(payload).is_err() {
                    self.cancel.cancel();
                    return;
                }
                if let Err(err) = self.queue.delete(&self.queue_name, id).await {
                    tracing::debug!(error = %err, queue = %self.queue_name, msg_id = id, "cable_pgmq: delete failed");
                }
            }
        }
    }
}

fn extract_payload(envelope: &Value) -> Bytes {
    let inner = envelope.get("p").and_then(Value::as_str).unwrap_or("");
    if let Some(b64_part) = inner.strip_prefix("base64:") {
        if let Ok(decoded) = b64_decode(b64_part) {
            return Bytes::from(decoded);
        }
    }
    Bytes::from(inner.to_owned().into_bytes())
}

struct PgMqDropGuard {
    cancel: CancellationToken,
    pool: PgPool,
    queue: PGMQueueExt,
    queue_name: String,
}

impl Drop for PgMqDropGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
        let pool = self.pool.clone();
        let queue = self.queue.clone();
        let queue_name = self.queue_name.clone();
        tokio::spawn(async move {
            let _ = sqlx::query("DELETE FROM loco_cable_pgmq_subs WHERE queue_name = $1")
                .bind(&queue_name)
                .execute(&pool)
                .await;
            let _ = queue.drop_queue(&queue_name).await;
        });
    }
}

/// Build a [`PubSub`] from configuration.
///
/// # Errors
/// Returns an error if the Postgres pool can't be created, the pgmq
/// extension can't be initialized, or the schema migration fails.
pub async fn create_provider(cfg: &PgMQCableConfig) -> Result<Arc<dyn PubSub>> {
    let provider = PgMqPubSub::connect(cfg).await?;
    Ok(provider as Arc<dyn PubSub>)
}

/// Tiny base64 encode/decode helpers — kept inline to avoid pulling in a
/// dedicated base64 crate. URL-safe alphabet, no padding.
fn b64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() * 4 + 2) / 3);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        }
    }
    out
}

fn b64_decode(s: &str) -> std::result::Result<Vec<u8>, ()> {
    fn val(c: u8) -> std::result::Result<u8, ()> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'-' => Ok(62),
            b'_' => Ok(63),
            _ => Err(()),
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let v0 = val(chunk[0])?;
        let v1 = val(*chunk.get(1).ok_or(())?)?;
        let v2 = chunk.get(2).map(|c| val(*c)).transpose()?;
        let v3 = chunk.get(3).map(|c| val(*c)).transpose()?;
        out.push((v0 << 2) | (v1 >> 4));
        if let Some(v2) = v2 {
            out.push(((v1 & 0x0f) << 4) | (v2 >> 2));
            if let Some(v3) = v3 {
                out.push(((v2 & 0x03) << 6) | v3);
            }
        }
    }
    Ok(out)
}
