//! OpenTelemetry metrics integration (feature `otel`).
//!
//! This wires up an OpenTelemetry [`SdkMeterProvider`] backed by an
//! [`opentelemetry-prometheus`](https://crates.io/crates/opentelemetry-prometheus)
//! pull exporter, and a
//! [`tracing-opentelemetry`](https://crates.io/crates/tracing-opentelemetry)
//! [`MetricsLayer`](tracing_opentelemetry::MetricsLayer) that bridges `tracing`
//! metric events to that provider. The collected metrics are exposed on the
//! existing `/_metrics` endpoint in the Prometheus text exposition format.
//!
//! With the `otel` feature enabled, Loco:
//!
//! 1. Initializes the meter provider + Prometheus registry once at startup
//!    ([`init`], called from the logger and boot paths).
//! 2. Installs the tracing [`MetricsLayer`](tracing_opentelemetry::MetricsLayer)
//!    into the subscriber, so application code can emit metrics with the
//!    `tracing` event convention and have them recorded by OpenTelemetry:
//!
//!    ```rust,ignore
//!    tracing::info!(monotonic_counter.orders_placed = 1_u64, tier = "pro");
//!    tracing::info!(histogram.job_duration_seconds = elapsed.as_secs_f64());
//!    ```
//! 3. Renders the Prometheus registry on `/_metrics` ([`render`]).
//!
//! For HTTP request metrics, attach [`track`] as middleware — it records to
//! native OpenTelemetry instruments (so it is unaffected by log-level
//! filtering), and the series show up on `/_metrics` automatically:
//!
//! ```rust,ignore
//! async fn after_routes(router: AxumRouter, ctx: &AppContext) -> Result<AxumRouter> {
//!     let http = loco_rs::metrics::otel::HttpMetrics::install(ctx);
//!     Ok(router.layer(axum::middleware::from_fn_with_state(
//!         http,
//!         loco_rs::metrics::otel::track,
//!     )))
//! }
//! ```

use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use axum::{
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use opentelemetry::{
    global,
    metrics::{Counter, Histogram, UpDownCounter},
    KeyValue,
};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use prometheus::{Encoder, Registry as PromRegistry, TextEncoder};
use tracing_subscriber::{Layer, Registry as TracingRegistry};

use crate::{app::AppContext, Error, Result};

static PROM_REGISTRY: OnceLock<PromRegistry> = OnceLock::new();
static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();

/// Meter name used for Loco's own instruments.
const METER_NAME: &str = "loco_rs";

/// Initialize the OpenTelemetry meter provider with a Prometheus pull exporter
/// and register it as the global provider.
///
/// Idempotent: subsequent calls are a no-op once initialized.
///
/// # Errors
/// Returns an error if the Prometheus exporter cannot be built.
pub fn init() -> Result<()> {
    if METER_PROVIDER.get().is_some() {
        return Ok(());
    }

    let registry = PromRegistry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .map_err(|err| Error::string(&err.to_string()))?;

    let provider = SdkMeterProvider::builder().with_reader(exporter).build();
    global::set_meter_provider(provider.clone());

    let _ = PROM_REGISTRY.set(registry);
    let _ = METER_PROVIDER.set(provider);

    Ok(())
}

/// The global meter provider, if [`init`] has run.
#[must_use]
pub fn meter_provider() -> Option<SdkMeterProvider> {
    METER_PROVIDER.get().cloned()
}

/// A `tracing` layer that bridges metric events (`monotonic_counter.*`,
/// `counter.*`, `histogram.*`, `gauge.*`) to the OpenTelemetry meter provider.
///
/// Returns `None` if [`init`] has not run yet.
#[must_use]
pub fn tracing_layer() -> Option<Box<dyn Layer<TracingRegistry> + Send + Sync>> {
    let provider = METER_PROVIDER.get()?.clone();
    Some(tracing_opentelemetry::MetricsLayer::new(provider).boxed())
}

/// Render the collected OpenTelemetry metrics in Prometheus text exposition
/// format. Returns `None` if [`init`] has not run.
#[must_use]
pub fn render() -> Option<String> {
    let registry = PROM_REGISTRY.get()?;
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    if encoder.encode(&registry.gather(), &mut buf).is_err() {
        return None;
    }
    String::from_utf8(buf).ok()
}

/// HTTP request metrics recorded through OpenTelemetry instruments.
///
/// Records, labeled by request method, matched route path, and status:
/// `http_server_requests` (counter), `http_server_request_duration_seconds`
/// (histogram), and `http_server_active_requests` (up/down counter).
pub struct HttpMetrics {
    requests: Counter<u64>,
    duration: Histogram<f64>,
    in_flight: UpDownCounter<i64>,
}

impl Default for HttpMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpMetrics {
    /// Create the HTTP instruments on the global meter.
    ///
    /// [`init`] should have run first so the instruments register with the
    /// Prometheus exporter.
    #[must_use]
    pub fn new() -> Self {
        let meter = global::meter(METER_NAME);
        Self {
            requests: meter
                .u64_counter("http_server_requests")
                .with_description("Total number of HTTP requests handled.")
                .build(),
            duration: meter
                .f64_histogram("http_server_request_duration_seconds")
                .with_description("HTTP request latencies in seconds.")
                .with_unit("s")
                .build(),
            in_flight: meter
                .i64_up_down_counter("http_server_active_requests")
                .with_description("Number of in-flight HTTP requests.")
                .build(),
        }
    }

    /// Create the collector and store it (as `Arc<HttpMetrics>`) in the
    /// application shared store, returning the `Arc` to attach [`track`] with.
    #[must_use]
    pub fn install(ctx: &AppContext) -> Arc<Self> {
        let collector = Arc::new(Self::new());
        ctx.shared_store.insert(collector.clone());
        collector
    }

    fn record(&self, method: &str, path: &str, status: u16, elapsed: Duration) {
        let attrs = [
            KeyValue::new("method", method.to_owned()),
            KeyValue::new("path", path.to_owned()),
            KeyValue::new("status", i64::from(status)),
        ];
        self.requests.add(1, &attrs);
        self.duration.record(elapsed.as_secs_f64(), &attrs);
    }
}

/// Axum middleware that records request metrics into the [`HttpMetrics`]
/// collector supplied as state.
///
/// Attach with
/// `router.layer(axum::middleware::from_fn_with_state(collector, track))`.
pub async fn track(State(metrics): State<Arc<HttpMetrics>>, req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_owned();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|matched| matched.as_str().to_owned())
        .unwrap_or_else(|| req.uri().path().to_owned());

    let in_flight_attrs = [
        KeyValue::new("method", method.clone()),
        KeyValue::new("path", path.clone()),
    ];
    metrics.in_flight.add(1, &in_flight_attrs);

    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed();

    metrics.in_flight.add(-1, &in_flight_attrs);
    metrics.record(&method, &path, response.status().as_u16(), elapsed);

    response
}

#[cfg(test)]
mod tests {
    use super::{init, render, HttpMetrics};
    use std::time::Duration;

    #[tokio::test]
    async fn init_records_and_renders() {
        init().expect("otel init");

        let collector = HttpMetrics::new();
        collector.record("GET", "/hello/{name}", 200, Duration::from_millis(5));

        let out = render().expect("rendered metrics");
        assert!(
            out.contains("http_server_requests"),
            "expected request counter in output, got:\n{out}"
        );
    }
}
