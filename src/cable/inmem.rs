//! Process-local pub/sub backend backed by `tokio::sync::broadcast`.
//!
//! Suitable for single-node dev / tests. For multi-process or multi-node
//! deployments choose [`super::redis`], [`super::pg`], [`super::sqlt`], or
//! [`super::pgmq`].

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::{broadcast, mpsc};

use crate::{cable::pubsub::Subscription, Result};

const BROADCAST_CAPACITY: usize = 256;

#[derive(Default)]
struct Inner {
    topics: DashMap<String, broadcast::Sender<Bytes>>,
}

impl Inner {
    fn sender(&self, topic: &str) -> broadcast::Sender<Bytes> {
        if let Some(s) = self.topics.get(topic) {
            return s.clone();
        }
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        self.topics
            .entry(topic.to_string())
            .or_insert_with(|| tx)
            .clone()
    }
}

/// Process-local pub/sub provider.
#[derive(Default, Clone)]
pub struct InMemPubSub {
    inner: Arc<Inner>,
}

#[async_trait]
impl super::PubSub for InMemPubSub {
    async fn publish(&self, topic: &str, payload: Bytes) -> Result<()> {
        let sender = self.inner.sender(topic);
        // It's fine if there are no subscribers; broadcast::send returns Err
        // in that case but it's not an application error.
        let _ = sender.send(payload);
        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Subscription> {
        let sender = self.inner.sender(topic);
        let mut rx = sender.subscribe();
        let (forward_tx, forward_rx) = mpsc::unbounded_channel::<Bytes>();
        let topic_owned = topic.to_string();
        let inner = self.inner.clone();

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(payload) => {
                        if forward_tx.send(payload).is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(topic = %topic_owned, lagged = n, "InMem subscriber lagged");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            // If this was the last subscriber, drop the topic to reclaim memory.
            if let Some(entry) = inner.topics.get(&topic_owned) {
                if entry.receiver_count() == 0 {
                    drop(entry);
                    inner.topics.remove(&topic_owned);
                }
            }
        });

        Ok(Subscription::new(forward_rx, ()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cable::PubSub;

    #[tokio::test]
    async fn publish_subscribe_roundtrip() {
        let bus = InMemPubSub::default();
        let mut sub = bus.subscribe("t1").await.unwrap();
        // Tiny pause to let the forwarder task install its broadcast receiver
        // before we publish.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        bus.publish("t1", Bytes::from_static(b"hello")).await.unwrap();
        let got = tokio::time::timeout(std::time::Duration::from_secs(1), sub.recv())
            .await
            .unwrap();
        assert_eq!(got.as_deref(), Some(&b"hello"[..]));
    }

    #[tokio::test]
    async fn isolated_topics() {
        let bus = InMemPubSub::default();
        let mut a = bus.subscribe("a").await.unwrap();
        let mut b = bus.subscribe("b").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        bus.publish("a", Bytes::from_static(b"x")).await.unwrap();
        let got_a = tokio::time::timeout(std::time::Duration::from_millis(200), a.recv())
            .await
            .unwrap();
        assert_eq!(got_a.as_deref(), Some(&b"x"[..]));
        // b should not receive
        let got_b = tokio::time::timeout(std::time::Duration::from_millis(50), b.recv()).await;
        assert!(got_b.is_err(), "topic b should not receive topic a's message");
    }
}
