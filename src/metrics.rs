//! Prometheus-format metrics for the `/_metrics` monitoring endpoint.
//!
//! Loco does not pull in a metrics backend of its own. Instead, the
//! [`/_metrics`](crate::controller::monitoring::metrics) endpoint renders a
//! small set of always-available runtime metrics in the
//! [Prometheus text exposition format][exposition] and appends whatever an
//! application chooses to expose through
//! [`Hooks::metrics`](crate::app::Hooks::metrics). This keeps the framework
//! dependency-free while letting applications wire in a real metrics registry
//! (e.g. the `metrics` or `prometheus` crates) and render it into the same
//! endpoint.
//!
//! The core metrics are:
//!
//! - `loco_build_info{...}` — a labeled gauge (always `1`) carrying the
//!   application/framework versions, build profile, target and environment.
//! - `loco_routes_total` — number of registered routes.
//! - `loco_uptime_seconds` — seconds since the server booted.
//! - `loco_start_time_seconds` — unix timestamp of when the server booted.
//!
//! [exposition]: https://prometheus.io/docs/instrumenting/exposition_formats/

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
fn escape_label(value: &str) -> String {
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
            let _ = writeln!(
                out,
                "loco_start_time_seconds {}",
                since_epoch.as_secs_f64()
            );
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
