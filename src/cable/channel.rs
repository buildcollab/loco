//! Action-Cable-style [`Channel`] trait + [`ChannelRegistry`].
//!
//! A `Channel` is a typed group of WebSocket / SSE endpoints. Implementors
//! decide which topics a connection streams from based on connection-time
//! `Params`, and optionally handle inbound client messages.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use serde::de::DeserializeOwned;

use crate::{app::AppContext, Result};

/// User-facing channel definition.
///
/// Implementors describe:
/// - which topics to subscribe to on connect ([`Channel::subscribed`]),
/// - what to do with inbound client messages ([`Channel::received`], optional),
/// - cleanup ([`Channel::unsubscribed`], optional).
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    /// Connection-time parameters parsed from the request (typically the
    /// query string). Use `serde_json::Value` if you don't care.
    type Params: DeserializeOwned + Send + Sync;

    /// Called once after a client connects. Return the list of pub/sub
    /// topics to stream payloads from. Returning an empty `Vec` is valid —
    /// the connection stays open but receives no broadcasts (useful for
    /// inbound-only channels).
    ///
    /// # Errors
    /// Any error here closes the connection.
    async fn subscribed(&self, ctx: &AppContext, params: Self::Params) -> Result<Vec<String>>;

    /// Optional hook for inbound client messages (WebSocket only). Default
    /// implementation drops the payload.
    async fn received(&self, _ctx: &AppContext, _data: Bytes) -> Result<()> {
        Ok(())
    }

    /// Optional cleanup hook called once when the connection closes.
    async fn unsubscribed(&self, _ctx: &AppContext) -> Result<()> {
        Ok(())
    }
}

/// Type-erased version of [`Channel`] used by the registry / transport.
///
/// You normally don't implement this directly — use the blanket impl by
/// implementing [`Channel`].
#[async_trait]
pub trait DynChannel: Send + Sync + 'static {
    /// Parse `params_raw` (a JSON value), then delegate to the concrete
    /// `subscribed`.
    async fn subscribed_dyn(
        &self,
        ctx: &AppContext,
        params_raw: serde_json::Value,
    ) -> Result<Vec<String>>;
    async fn received_dyn(&self, ctx: &AppContext, data: Bytes) -> Result<()>;
    async fn unsubscribed_dyn(&self, ctx: &AppContext) -> Result<()>;
}

#[async_trait]
impl<C: Channel> DynChannel for C {
    async fn subscribed_dyn(
        &self,
        ctx: &AppContext,
        params_raw: serde_json::Value,
    ) -> Result<Vec<String>> {
        let params: C::Params = serde_json::from_value(params_raw).map_err(crate::Error::JSON)?;
        self.subscribed(ctx, params).await
    }
    async fn received_dyn(&self, ctx: &AppContext, data: Bytes) -> Result<()> {
        self.received(ctx, data).await
    }
    async fn unsubscribed_dyn(&self, ctx: &AppContext) -> Result<()> {
        self.unsubscribed(ctx).await
    }
}

/// Registry of named channels populated by `Hooks::register_channels`.
///
/// Each entry is keyed by a stable name (e.g. `"chat"`) which is also the
/// route segment a transport handler looks up.
#[derive(Default, Clone)]
pub struct ChannelRegistry {
    channels: HashMap<String, Arc<dyn DynChannel>>,
}

impl ChannelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a channel under `name`. Replaces any prior registration.
    pub fn register<C: Channel>(&mut self, name: impl Into<String>, channel: C) -> &mut Self {
        self.channels.insert(name.into(), Arc::new(channel));
        self
    }

    /// Look up a channel by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn DynChannel>> {
        self.channels.get(name).cloned()
    }

    /// Iterate over all registered channel names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.channels.keys().map(String::as_str)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }
}

impl std::fmt::Debug for ChannelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelRegistry")
            .field("channels", &self.channels.keys().collect::<Vec<_>>())
            .finish()
    }
}
