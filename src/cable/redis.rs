//! Redis pub/sub backend.
//!
//! Uses Redis' native `PUBLISH` / `SUBSCRIBE`. Multi-node out of the box, no
//! polling. Messages are not persisted — subscribers only receive what is
//! published while they are connected (matching standard Redis pub/sub
//! semantics).

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use redis::{aio::MultiplexedConnection, AsyncCommands, Client};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    cable::pubsub::{PubSub, Subscription},
    config::RedisCableConfig,
    Error, Result,
};

pub struct RedisPubSub {
    client: Client,
    /// A multiplexed connection used for `PUBLISH` only — subscribe needs a
    /// dedicated connection.
    publisher: tokio::sync::Mutex<MultiplexedConnection>,
}

impl RedisPubSub {
    pub async fn connect(cfg: &RedisCableConfig) -> Result<Arc<Self>> {
        let client = Client::open(cfg.uri.as_str()).map_err(Error::wrap)?;
        let publisher = client
            .get_multiplexed_async_connection()
            .await
            .map_err(Error::wrap)?;
        Ok(Arc::new(Self {
            client,
            publisher: tokio::sync::Mutex::new(publisher),
        }))
    }
}

#[async_trait]
impl PubSub for RedisPubSub {
    async fn publish(&self, topic: &str, payload: Bytes) -> Result<()> {
        let mut conn = self.publisher.lock().await;
        let _: i64 = conn
            .publish(topic, payload.as_ref())
            .await
            .map_err(Error::wrap)?;
        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Subscription> {
        // Each subscription gets its own dedicated connection — Redis
        // requires this.
        let pubsub_conn = self.client.get_async_pubsub().await.map_err(Error::wrap)?;
        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
        let topic_owned = topic.to_string();
        let cancel_for_task = cancel.clone();

        tokio::spawn(async move {
            let mut pubsub = pubsub_conn;
            if let Err(err) = pubsub.subscribe(&topic_owned).await {
                tracing::warn!(error = %err, topic = %topic_owned, "cable_redis: subscribe failed");
                return;
            }
            let mut stream = pubsub.on_message();
            loop {
                tokio::select! {
                    () = cancel_for_task.cancelled() => break,
                    msg = stream.next() => match msg {
                        Some(m) => {
                            let payload: Vec<u8> = match m.get_payload() {
                                Ok(p) => p,
                                Err(err) => {
                                    tracing::warn!(error = %err, "cable_redis: bad payload");
                                    continue;
                                }
                            };
                            if tx.send(Bytes::from(payload)).is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
            }
        });

        Ok(Subscription::new(rx, RedisDropGuard { cancel }))
    }
}

struct RedisDropGuard {
    cancel: CancellationToken,
}

impl Drop for RedisDropGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Build a [`PubSub`] from configuration.
///
/// # Errors
/// Returns an error if the Redis client / connection can't be established.
pub async fn create_provider(cfg: &RedisCableConfig) -> Result<Arc<dyn PubSub>> {
    let provider = RedisPubSub::connect(cfg).await?;
    Ok(provider as Arc<dyn PubSub>)
}
