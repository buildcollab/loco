//! Server information manifest served by the `/_server` monitoring endpoint.
//!
//! The manifest combines three tiers of data that together describe a running
//! server:
//!
//! 1. **Build-time metadata** ([`BuildInfo`]) — captured by the build script
//!    (`build.rs`) and baked into the binary. This covers things that can only
//!    be known at compile time: the framework version, the `rustc` version, the
//!    build profile, and the target triple.
//! 2. **Boot-time metadata** — the registered [`routes`](ServerInfo::routes),
//!    application name, version, and environment. Routes are only known once
//!    [`Hooks::routes`](crate::app::Hooks::routes) has run, but never change
//!    afterwards, so they are collected once during boot.
//! 3. **User-provided fields** — arbitrary JSON returned from
//!    [`Hooks::server_info_extras`](crate::app::Hooks::server_info_extras),
//!    letting applications surface their own information.
//!
//! The assembled [`ServerInfo`] is stored in the application
//! [`SharedStore`](crate::app::SharedStore) during boot and rendered as JSON by
//! [`crate::controller::monitoring::server_info`].

use serde::Serialize;

use crate::{app::AppContext, controller::AppRoutes};

/// Build-time metadata captured by the build script (`build.rs`).
#[derive(Debug, Clone, Serialize)]
pub struct BuildInfo {
    /// Version of the `loco-rs` framework the application was built against.
    pub loco_version: &'static str,
    /// The `rustc` version used to compile the application.
    pub rustc_version: &'static str,
    /// Cargo build profile (e.g. `debug` or `release`).
    pub profile: &'static str,
    /// Target triple the application was compiled for.
    pub target: &'static str,
}

impl BuildInfo {
    /// Return the build metadata baked into this binary at compile time.
    #[must_use]
    pub const fn current() -> Self {
        Self {
            loco_version: env!("CARGO_PKG_VERSION"),
            rustc_version: env!("LOCO_BUILD_RUSTC_VERSION"),
            profile: env!("LOCO_BUILD_PROFILE"),
            target: env!("LOCO_BUILD_TARGET"),
        }
    }
}

/// A single registered route entry.
#[derive(Debug, Clone, Serialize)]
pub struct RouteInfo {
    /// HTTP methods handled at this path (e.g. `GET`, `POST`).
    pub methods: Vec<String>,
    /// The normalized URI path.
    pub uri: String,
}

/// Full server manifest returned by the `/_server` endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ServerInfo {
    /// Application crate name (from [`Hooks::app_name`](crate::app::Hooks::app_name)).
    pub name: String,
    /// Application version (from [`Hooks::app_version`](crate::app::Hooks::app_version)).
    pub version: String,
    /// Environment the server is running in (e.g. `development`, `production`).
    pub environment: String,
    /// Build-time metadata.
    pub build: BuildInfo,
    /// Registered routes, captured at boot.
    pub routes: Vec<RouteInfo>,
    /// Arbitrary application-provided fields
    /// (from [`Hooks::server_info_extras`](crate::app::Hooks::server_info_extras)).
    /// Omitted from the JSON output when `null`.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub custom: serde_json::Value,
}

impl ServerInfo {
    /// Assemble the full manifest from the application [`Hooks`](crate::app::Hooks)
    /// and its registered routes.
    #[must_use]
    pub fn from_hooks<H: crate::app::Hooks>(ctx: &AppContext, app_routes: &AppRoutes) -> Self {
        Self {
            name: H::app_name().to_string(),
            version: H::app_version(),
            environment: ctx.environment.to_string(),
            build: BuildInfo::current(),
            routes: Self::routes_from(app_routes),
            custom: H::server_info_extras(ctx),
        }
    }

    /// Collect the route list from an [`AppRoutes`] instance.
    #[must_use]
    pub fn routes_from(app_routes: &AppRoutes) -> Vec<RouteInfo> {
        app_routes
            .collect()
            .into_iter()
            .map(|route| RouteInfo {
                methods: route.actions.iter().map(ToString::to_string).collect(),
                uri: route.uri,
            })
            .collect()
    }

    /// A minimal manifest used as a fallback when the full manifest has not been
    /// populated into the [`SharedStore`](crate::app::SharedStore) (for example,
    /// when the endpoint is exercised outside of the normal boot flow).
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            name: "unknown".to_string(),
            version: "unknown".to_string(),
            environment: "unknown".to_string(),
            build: BuildInfo::current(),
            routes: Vec::new(),
            custom: serde_json::Value::Null,
        }
    }
}
