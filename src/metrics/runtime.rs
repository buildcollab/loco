//! Tokio runtime metrics helper.
//!
//! [`render`] returns the current Tokio runtime's metrics in the Prometheus
//! text exposition format, intended to be included from an application's
//! [`Hooks::metrics`](crate::app::Hooks::metrics) implementation:
//!
//! ```rust,ignore
//! fn metrics(_ctx: &AppContext) -> String {
//!     loco_rs::metrics::runtime::render()
//! }
//! ```
//!
//! Only the **stable** subset of Tokio's
//! [`RuntimeMetrics`](tokio::runtime::RuntimeMetrics) is used, so this works on
//! a normal build with no extra flags. For the richer per-worker counters
//! (poll counts, busy durations, steal counts, …) use the
//! [`tokio-metrics`](https://crates.io/crates/tokio-metrics) crate — its
//! `RuntimeMonitor` exposes them, but it requires building with
//! `RUSTFLAGS="--cfg tokio_unstable"`. You can render a `tokio-metrics`
//! interval and append it here in the same way.

use std::fmt::Write;

/// Render the current Tokio runtime's metrics in Prometheus text exposition
/// format.
///
/// Returns an empty string when called outside of a Tokio runtime context
/// (there is nothing to report).
#[must_use]
pub fn render() -> String {
    let mut out = String::new();

    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return out;
    };
    let metrics = handle.metrics();

    let _ = writeln!(
        out,
        "# HELP loco_runtime_workers Number of worker threads used by the Tokio runtime."
    );
    let _ = writeln!(out, "# TYPE loco_runtime_workers gauge");
    let _ = writeln!(out, "loco_runtime_workers {}", metrics.num_workers());

    let _ = writeln!(
        out,
        "# HELP loco_runtime_alive_tasks Current number of alive tasks in the Tokio runtime."
    );
    let _ = writeln!(out, "# TYPE loco_runtime_alive_tasks gauge");
    let _ = writeln!(
        out,
        "loco_runtime_alive_tasks {}",
        metrics.num_alive_tasks()
    );

    let _ = writeln!(
        out,
        "# HELP loco_runtime_global_queue_depth Number of tasks in the runtime's global queue."
    );
    let _ = writeln!(out, "# TYPE loco_runtime_global_queue_depth gauge");
    let _ = writeln!(
        out,
        "loco_runtime_global_queue_depth {}",
        metrics.global_queue_depth()
    );

    out
}

#[cfg(test)]
mod tests {
    use super::render;

    #[tokio::test]
    async fn renders_runtime_metrics() {
        let out = render();
        assert!(out.contains("loco_runtime_workers "));
        assert!(out.contains("loco_runtime_alive_tasks "));
        assert!(out.contains("loco_runtime_global_queue_depth "));
    }
}
