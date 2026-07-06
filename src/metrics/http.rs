//! HTTP request metrics helper.
//!
//! A small, dependency-free collector for HTTP request metrics, exposed as an
//! Axum middleware plus a Prometheus renderer for
//! [`Hooks::metrics`](crate::app::Hooks::metrics). It records, labeled by
//! request method, matched route path, and response status:
//!
//! - `loco_http_requests_total` — a counter of handled requests.
//! - `loco_http_request_duration_seconds` — a latency histogram
//!   (`_bucket` / `_sum` / `_count`).
//! - `loco_http_requests_in_flight` — a gauge of requests currently being
//!   served.
//!
//! The route label uses Axum's [`MatchedPath`] (the route *template*, e.g.
//! `/users/{id}`) when available, keeping label cardinality bounded rather than
//! exploding on every concrete URL.
//!
//! # Usage
//!
//! Install the collector and middleware in your `Hooks::after_routes`, and
//! render it from `Hooks::metrics`:
//!
//! ```rust,ignore
//! async fn after_routes(router: AxumRouter, ctx: &AppContext) -> Result<AxumRouter> {
//!     let http = loco_rs::metrics::http::HttpMetrics::install(ctx);
//!     Ok(router.layer(axum::middleware::from_fn_with_state(
//!         http,
//!         loco_rs::metrics::http::track,
//!     )))
//! }
//!
//! fn metrics(ctx: &AppContext) -> String {
//!     loco_rs::metrics::http::render(ctx)
//! }
//! ```

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use dashmap::DashMap;

use crate::{app::AppContext, metrics::escape_label};

/// Upper bounds (in seconds) for the request-duration histogram buckets.
///
/// Matches the Prometheus client default buckets.
const BUCKETS: [f64; 11] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Accumulated stats for a single `(method, path, status)` label set.
struct Series {
    /// One counter per histogram bucket, plus a final `+Inf` slot.
    buckets: [AtomicU64; BUCKETS.len() + 1],
    /// Total number of observations (equals the `+Inf` cumulative count).
    count: AtomicU64,
    /// Sum of observed durations, in nanoseconds.
    sum_nanos: AtomicU64,
}

impl Series {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_nanos: AtomicU64::new(0),
        }
    }

    fn observe(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        let idx = BUCKETS
            .iter()
            .position(|bound| seconds <= *bound)
            .unwrap_or(BUCKETS.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        // Saturating cast: only overflows for durations of hundreds of years.
        self.sum_nanos.fetch_add(
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }
}

/// Collector for HTTP request metrics.
///
/// Create and register one with [`HttpMetrics::install`], attach [`track`] as
/// middleware, and expose it via [`render`].
#[derive(Default)]
pub struct HttpMetrics {
    series: DashMap<(String, String, u16), Series>,
    in_flight: AtomicU64,
}

impl HttpMetrics {
    /// Create a new, empty collector.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Create a collector and store it (as `Arc<HttpMetrics>`) in the
    /// application [`SharedStore`](crate::app::SharedStore) so it can later be
    /// rendered with [`render`]. Returns the `Arc` to attach the [`track`]
    /// middleware with.
    #[must_use]
    pub fn install(ctx: &AppContext) -> Arc<Self> {
        let collector = Self::new();
        ctx.shared_store.insert(collector.clone());
        collector
    }

    /// Record a single completed request.
    pub fn record(&self, method: &str, path: &str, status: u16, elapsed: Duration) {
        self.series
            .entry((method.to_owned(), path.to_owned(), status))
            .or_insert_with(Series::new)
            .observe(elapsed);
    }

    /// Render the collected metrics in Prometheus text exposition format.
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();

        let _ = writeln!(
            out,
            "# HELP loco_http_requests_total Total number of HTTP requests handled."
        );
        let _ = writeln!(out, "# TYPE loco_http_requests_total counter");
        for entry in &self.series {
            let (method, path, status) = entry.key();
            let _ = writeln!(
                out,
                "loco_http_requests_total{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}",
                escape_label(method),
                escape_label(path),
                status,
                entry.count.load(Ordering::Relaxed),
            );
        }

        let _ = writeln!(
            out,
            "# HELP loco_http_request_duration_seconds HTTP request latencies in seconds."
        );
        let _ = writeln!(out, "# TYPE loco_http_request_duration_seconds histogram");
        for entry in &self.series {
            let (method, path, status) = entry.key();
            let labels = format!(
                "method=\"{}\",path=\"{}\",status=\"{}\"",
                escape_label(method),
                escape_label(path),
                status,
            );

            let mut cumulative = 0u64;
            for (i, bound) in BUCKETS.iter().enumerate() {
                cumulative += entry.buckets[i].load(Ordering::Relaxed);
                let _ = writeln!(
                    out,
                    "loco_http_request_duration_seconds_bucket{{{labels},le=\"{bound}\"}} {cumulative}"
                );
            }
            cumulative += entry.buckets[BUCKETS.len()].load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "loco_http_request_duration_seconds_bucket{{{labels},le=\"+Inf\"}} {cumulative}"
            );
            let sum_seconds = entry.sum_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
            let _ = writeln!(
                out,
                "loco_http_request_duration_seconds_sum{{{labels}}} {sum_seconds}"
            );
            let _ = writeln!(
                out,
                "loco_http_request_duration_seconds_count{{{labels}}} {}",
                entry.count.load(Ordering::Relaxed),
            );
        }

        let _ = writeln!(
            out,
            "# HELP loco_http_requests_in_flight Number of HTTP requests currently being served."
        );
        let _ = writeln!(out, "# TYPE loco_http_requests_in_flight gauge");
        let _ = writeln!(
            out,
            "loco_http_requests_in_flight {}",
            self.in_flight.load(Ordering::Relaxed)
        );

        out
    }
}

/// Axum middleware that records request metrics into the [`HttpMetrics`]
/// collector supplied as state.
///
/// Attach it with
/// `router.layer(axum::middleware::from_fn_with_state(collector, track))`.
pub async fn track(State(metrics): State<Arc<HttpMetrics>>, req: Request, next: Next) -> Response {
    metrics.in_flight.fetch_add(1, Ordering::Relaxed);

    let method = req.method().as_str().to_owned();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|matched| matched.as_str().to_owned())
        .unwrap_or_else(|| req.uri().path().to_owned());

    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed();

    metrics.record(&method, &path, response.status().as_u16(), elapsed);
    metrics.in_flight.fetch_sub(1, Ordering::Relaxed);

    response
}

/// Render the [`HttpMetrics`] previously registered with [`HttpMetrics::install`]
/// from the application shared store. Returns an empty string if none was
/// installed.
#[must_use]
pub fn render(ctx: &AppContext) -> String {
    ctx.shared_store
        .get_ref::<Arc<HttpMetrics>>()
        .map(|collector| collector.render())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{track, HttpMetrics};
    use axum::{routing::get, Router};
    use std::sync::Arc;
    use tower::ServiceExt;

    #[tokio::test]
    async fn records_requests_and_renders() {
        let collector = HttpMetrics::new();

        let app: Router = Router::new()
            .route("/hello/{name}", get(|| async { "hi" }))
            .layer(axum::middleware::from_fn_with_state(
                collector.clone(),
                track,
            ));

        for _ in 0..3 {
            let req = axum::http::Request::builder()
                .uri("/hello/loco")
                .body(axum::body::Body::empty())
                .unwrap();
            let res = app.clone().oneshot(req).await.unwrap();
            assert_eq!(res.status(), 200);
        }

        let out = Arc::clone(&collector).render();

        // Counter uses the matched route template, not the concrete path.
        assert!(out.contains(
            "loco_http_requests_total{method=\"GET\",path=\"/hello/{name}\",status=\"200\"} 3"
        ));
        // Histogram series is present.
        assert!(out.contains("loco_http_request_duration_seconds_bucket{"));
        assert!(out.contains("le=\"+Inf\"} 3"));
        assert!(out.contains("loco_http_request_duration_seconds_count{method=\"GET\",path=\"/hello/{name}\",status=\"200\"} 3"));
        // In-flight returns to zero once requests complete.
        assert!(out.contains("loco_http_requests_in_flight 0"));
    }
}
