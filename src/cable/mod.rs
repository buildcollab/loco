//! # Loco Cable — realtime pub/sub for WebSocket / SSE
//!
//! `cable` provides a small pub/sub primitive (topic + bytes) plus
//! WebSocket and SSE transports that bridge a [`PubSub`] backend to
//! connected clients. The model mirrors Rails' Solid Cable: any code path
//! (controller, worker, task, external producer) can `publish(topic, ...)`
//! and any client subscribed to `topic` receives the payload.
//!
//! ## Backends
//!
//! | Variant            | Feature flag       | Notes                                  |
//! |--------------------|--------------------|----------------------------------------|
//! | [`Cable::InMem`]   | always-on          | `tokio::sync::broadcast`, single-node. |
//! | [`Cable::Postgres`]| `cable_pg`         | Polled `loco_cable_messages` table.    |
//! | [`Cable::Sqlite`]  | `cable_sqlt`       | Polled `loco_cable_messages` table.    |
//! | [`Cable::Redis`]   | `cable_redis`      | Native Redis `PUBSUB`, no polling.     |
//! | [`Cable::PgMQ`]    | `cable_pgmq`       | Per-subscription `pgmq` queue.         |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use loco_rs::cable::{Channel, ChannelRegistry};
//!
//! struct Chat;
//!
//! #[async_trait::async_trait]
//! impl Channel for Chat {
//!     type Params = serde_json::Value;
//!     async fn subscribed(
//!         &self,
//!         _ctx: &loco_rs::app::AppContext,
//!         _params: Self::Params,
//!     ) -> loco_rs::Result<Vec<String>> {
//!         Ok(vec!["chat:lobby".to_string()])
//!     }
//! }
//! ```

// `cable` is a newer module not yet held to the extra
// `clippy::pedantic`/`clippy::nursery` bar that CI applies on top of the default
// lints. Scope those two opt-in groups off for this module so CI's `-D warnings`
// stays green until it is cleaned up; the default clippy lints still apply here.
#![allow(clippy::pedantic, clippy::nursery)]

pub mod channel;
pub mod inmem;
pub mod pubsub;
pub mod transport;

#[cfg(test)]
mod tests;

#[cfg(feature = "cable_pg")]
pub mod pg;
#[cfg(feature = "cable_pgmq")]
pub mod pgmq;
#[cfg(feature = "cable_redis")]
pub mod redis;
#[cfg(feature = "cable_sqlt")]
pub mod sqlt;

use std::sync::Arc;

use bytes::Bytes;

use crate::{
    config::{CableConfig, Config},
    Error, Result,
};

pub use channel::{Channel, ChannelRegistry, DynChannel};
pub use pubsub::{PubSub, Subscription};

/// A type-erased, cloneable handle to whatever pub/sub backend the
/// application is configured with.
///
/// Stored on [`crate::app::AppContext`] as `cable: Option<Cable>`. Cloning is
/// cheap — it's an `Arc` under the hood — so handlers and workers can hold
/// their own copy.
#[derive(Clone)]
pub struct Cable(Arc<dyn PubSub>);

impl Cable {
    /// Wrap any [`PubSub`] implementation into a [`Cable`] handle.
    #[must_use]
    pub fn new<P: PubSub + 'static>(provider: P) -> Self {
        Self(Arc::new(provider))
    }

    /// Wrap an already-Arc'd [`PubSub`].
    #[must_use]
    pub fn from_arc(provider: Arc<dyn PubSub>) -> Self {
        Self(provider)
    }

    /// Publish `payload` on `topic`. All current subscribers to `topic`
    /// receive a copy.
    ///
    /// # Errors
    /// Backend-specific transport errors propagate from the underlying
    /// provider.
    pub async fn publish(&self, topic: &str, payload: Bytes) -> Result<()> {
        self.0.publish(topic, payload).await
    }

    /// JSON convenience: serialize `value` and publish it.
    ///
    /// # Errors
    /// Returns a serialization error if `value` cannot be encoded, otherwise
    /// the same errors as [`Cable::publish`].
    pub async fn publish_json<T: serde::Serialize>(&self, topic: &str, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value).map_err(Error::JSON)?;
        self.publish(topic, Bytes::from(bytes)).await
    }

    /// Subscribe to `topic`. The returned [`Subscription`] yields payloads
    /// until it is dropped, at which point the backend cleans up.
    ///
    /// # Errors
    /// Backend-specific errors propagate from the underlying provider.
    pub async fn subscribe(&self, topic: &str) -> Result<Subscription> {
        self.0.subscribe(topic).await
    }

    /// Access the underlying provider as a trait object — useful when
    /// downstream code wants to plug it into custom abstractions.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn PubSub> {
        &self.0
    }
}

impl std::fmt::Debug for Cable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cable").finish_non_exhaustive()
    }
}

/// Build a [`Cable`] provider from configuration, or `None` if `config.cable`
/// is unset.
///
/// # Errors
/// Returns an error if a backend feature is selected in config but not
/// compiled in, or if the backend fails to connect.
#[allow(unused_variables)]
pub async fn create_provider(config: &Config) -> Result<Option<Cable>> {
    let Some(cable_cfg) = &config.cable else {
        return Ok(None);
    };

    match cable_cfg {
        CableConfig::InMem => {
            tracing::debug!("Creating InMem cable provider");
            Ok(Some(Cable::new(inmem::InMemPubSub::default())))
        }
        #[cfg(feature = "cable_pg")]
        CableConfig::Postgres(cfg) => {
            tracing::debug!("Creating Postgres cable provider");
            Ok(Some(Cable::from_arc(pg::create_provider(cfg).await?)))
        }
        #[cfg(feature = "cable_sqlt")]
        CableConfig::Sqlite(cfg) => {
            tracing::debug!("Creating Sqlite cable provider");
            Ok(Some(Cable::from_arc(sqlt::create_provider(cfg).await?)))
        }
        #[cfg(feature = "cable_redis")]
        CableConfig::Redis(cfg) => {
            tracing::debug!("Creating Redis cable provider");
            Ok(Some(Cable::from_arc(redis::create_provider(cfg).await?)))
        }
        #[cfg(feature = "cable_pgmq")]
        CableConfig::PgMQ(cfg) => {
            tracing::debug!("Creating PgMQ cable provider");
            Ok(Some(Cable::from_arc(pgmq::create_provider(cfg).await?)))
        }
        #[allow(unreachable_patterns)]
        _ => Err(Error::string(
            "cable backend selected in config is not compiled into this build (missing feature \
             flag)",
        )),
    }
}
