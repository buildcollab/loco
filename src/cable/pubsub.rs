//! Core [`PubSub`] trait — the lowest-level cable primitive.

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::Result;

/// A pluggable pub/sub backend.
///
/// Implementations are stored on [`crate::app::AppContext`] as
/// `Arc<dyn PubSub>` (wrapped in [`super::Cable`]). The trait is dyn-safe
/// because [`Subscription`] is a concrete type.
#[async_trait]
pub trait PubSub: Send + Sync + 'static {
    /// Publish `payload` to all subscribers of `topic`.
    ///
    /// # Errors
    /// Backend-specific transport errors.
    async fn publish(&self, topic: &str, payload: Bytes) -> Result<()>;

    /// Subscribe to `topic`. Drop the returned [`Subscription`] to unsubscribe
    /// and free backend resources (broadcast slot, polling loop, ephemeral
    /// queue, etc.).
    ///
    /// # Errors
    /// Backend-specific transport errors.
    async fn subscribe(&self, topic: &str) -> Result<Subscription>;
}

/// A live subscription to a topic. Yields payloads via [`Self::recv`] and
/// cleans up automatically when dropped.
pub struct Subscription {
    rx: UnboundedReceiver<Bytes>,
    /// Held for its `Drop` impl — releases backend-side resources.
    _guard: Box<dyn Send + Sync>,
}

impl Subscription {
    /// Build a new [`Subscription`] from a channel receiver and an opaque
    /// drop-guard. Backends call this from their `subscribe` implementations.
    #[must_use]
    pub fn new<G: Send + Sync + 'static>(rx: UnboundedReceiver<Bytes>, guard: G) -> Self {
        Self {
            rx,
            _guard: Box::new(guard),
        }
    }

    /// Receive the next payload. Returns `None` once the backend closes the
    /// channel (e.g. on shutdown).
    pub async fn recv(&mut self) -> Option<Bytes> {
        self.rx.recv().await
    }

    /// Try to receive a payload without awaiting.
    pub fn try_recv(&mut self) -> Option<Bytes> {
        self.rx.try_recv().ok()
    }
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription").finish_non_exhaustive()
    }
}
