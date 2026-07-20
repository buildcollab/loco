//! Prometheus-format metrics for the `/_metrics` monitoring endpoint.
//!
//! Loco does not pull in a metrics backend of its own. Instead, the
//! [`/_metrics`](crate::controller::monitoring::metrics) endpoint renders a
//! small set of always-available runtime metrics in the
//! [Prometheus text exposition format][exposition] and appends whatever an
//! application chooses to expose through
//! [`Hooks::metrics`](crate::app::Hooks::metrics). This keeps the framework
//! dependency-free while letting applications opt into richer metrics.
//!
//! The core metrics (always emitted) are:
//!
//! - `loco_build_info{...}` — a labeled gauge (always `1`) carrying the
//!   application/framework versions, build profile, target and environment.
//! - `loco_routes_total` — number of registered routes.
//! - `loco_uptime_seconds` — seconds since the server booted.
//! - `loco_start_time_seconds` — unix timestamp of when the server booted.
//!
//! # Opt-in metric libraries
//!
//! Two dependency-free helpers are provided that an application can plug into
//! its [`Hooks::metrics`](crate::app::Hooks::metrics) implementation:
//!
//! - [`http`] — HTTP request metrics (request counts, a latency histogram, and
//!   an in-flight gauge) collected by a small Axum middleware.
//! - [`runtime`] — Tokio runtime metrics (worker count, alive tasks, global
//!   queue depth) read from the current runtime.
//!
//! ```rust,ignore
//! // In your `App` `Hooks` implementation:
//! async fn after_routes(router: AxumRouter, ctx: &AppContext) -> Result<AxumRouter> {
//!     // Install the HTTP metrics collector + middleware.
//!     let http = loco_rs::metrics::http::HttpMetrics::install(ctx);
//!     Ok(router.layer(axum::middleware::from_fn_with_state(
//!         http,
//!         loco_rs::metrics::http::track,
//!     )))
//! }
//!
//! fn metrics(ctx: &AppContext) -> String {
//!     let mut out = loco_rs::metrics::http::render(ctx);
//!     out.push_str(&loco_rs::metrics::runtime::render());
//!     out
//! }
//! ```
//!
//! [exposition]: https://prometheus.io/docs/instrumenting/exposition_formats/

pub mod http;
#[cfg(feature = "otel")]
pub mod otel;
pub mod runtime;

use std::{
    fmt::Write,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{app::AppContext, server_info::ServerInfo};

/// Records when the process booted so uptime can be computed per scrape.
///
/// Stored in the application [`SharedStore`](crate::app::SharedStore) during
/// boot.
#[derive(Debug, Clone, Copy)]
pub struct BootTime {
    /// Monotonic clock reading captured at boot (used for uptime).
    pub started: Instant,
    /// Wall-clock time captured at boot (used for the start timestamp).
    pub started_at: SystemTime,
}

impl BootTime {
    /// Capture the current time as the boot time.
    #[must_use]
    pub fn now() -> Self {
        Self {
            started: Instant::now(),
            started_at: SystemTime::now(),
        }
    }
}

impl Default for BootTime {
    fn default() -> Self {
        Self::now()
    }
}

/// A per-scrape renderer for application-provided metrics.
///
/// Populated at boot from [`Hooks::metrics`](crate::app::Hooks::metrics) and
/// invoked on every request to `/_metrics`, so the values it returns can be
/// fully dynamic. A plain function pointer is used (rather than a boxed closure)
/// so the value is `'static + Send + Sync` without constraining the boot-time
/// `Hooks` type parameter.
pub struct MetricsHook(pub fn(&AppContext) -> String);

/// Escape a string for use as a Prometheus label value.
pub(crate) fn escape_label(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

/// Render the core Loco metrics plus any application-provided metrics in
/// Prometheus text exposition format.
#[must_use]
pub fn render(ctx: &AppContext) -> String {
    let mut out = String::new();

    if let Some(info) = ctx.shared_store.get_ref::<ServerInfo>() {
        let _ = writeln!(
            out,
            "# HELP loco_build_info Build information about the running Loco application."
        );
        let _ = writeln!(out, "# TYPE loco_build_info gauge");
        let _ = writeln!(
            out,
            "loco_build_info{{version=\"{}\",loco_version=\"{}\",rustc_version=\"{}\",profile=\"{}\",target=\"{}\",environment=\"{}\"}} 1",
            escape_label(&info.version),
            escape_label(info.build.loco_version),
            escape_label(info.build.rustc_version),
            escape_label(info.build.profile),
            escape_label(info.build.target),
            escape_label(&info.environment),
        );

        let _ = writeln!(
            out,
            "# HELP loco_routes_total Number of routes registered in the application."
        );
        let _ = writeln!(out, "# TYPE loco_routes_total gauge");
        let _ = writeln!(out, "loco_routes_total {}", info.routes.len());
    }

    if let Some(boot) = ctx.shared_store.get_ref::<BootTime>() {
        let _ = writeln!(
            out,
            "# HELP loco_uptime_seconds Seconds since the server booted."
        );
        let _ = writeln!(out, "# TYPE loco_uptime_seconds gauge");
        let _ = writeln!(
            out,
            "loco_uptime_seconds {}",
            boot.started.elapsed().as_secs_f64()
        );

        if let Ok(since_epoch) = boot.started_at.duration_since(UNIX_EPOCH) {
            let _ = writeln!(
                out,
                "# HELP loco_start_time_seconds Unix timestamp of when the server booted."
            );
            let _ = writeln!(out, "# TYPE loco_start_time_seconds gauge");
            let _ = writeln!(out, "loco_start_time_seconds {}", since_epoch.as_secs_f64());
        }
    }

    // When the `otel` feature is enabled, append the OpenTelemetry-collected
    // metrics (HTTP instruments, `tracing` metric events, and any application
    // instruments on the global meter) rendered from the Prometheus registry.
    #[cfg(feature = "otel")]
    if let Some(text) = otel::render() {
        if !text.trim().is_empty() {
            out.push_str(text.trim_end());
            out.push('\n');
        }
    }

    if let Some(hook) = ctx.shared_store.get_ref::<MetricsHook>() {
        let extra = (hook.0)(ctx);
        if !extra.trim().is_empty() {
            out.push_str(extra.trim_end());
            out.push('\n');
        }
    }

    out
}
