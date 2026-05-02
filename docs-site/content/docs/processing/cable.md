+++
title = "Cable (Realtime)"
description = ""
date = 2026-05-02T00:00:00+00:00
updated = 2026-05-02T00:00:00+00:00
draft = false
weight = 6
sort_by = "weight"
template = "docs/page.html"

[extra]
lead = ""
toc = true
top = false
flair =[]
+++

Loco's `cable` module provides realtime pub/sub for WebSocket and SSE
clients — the equivalent of Rails' Solid Cable. Any code path (controller,
worker, task, external producer) can `publish(topic, payload)` and every
client subscribed to that topic receives the payload, in near-real-time.

## Backends

Pick a backend that matches your deployment shape. All five are off-the-shelf
and live behind feature flags (mirroring `bg_*`).

| Backend                  | Feature flag   | When to use it                                                     |
|--------------------------|----------------|---------------------------------------------------------------------|
| `CableConfig::InMem`     | always-on      | Single-process dev / tests. No external services.                   |
| `CableConfig::Postgres`  | `cable_pg`     | "No Redis required" multi-node. Polls `loco_cable_messages`.        |
| `CableConfig::Sqlite`    | `cable_sqlt`   | Single-host persistent (e.g. embedded). File-system bound.          |
| `CableConfig::Redis`     | `cable_redis`  | Native `PUBSUB`, no polling, multi-node fan-out.                    |
| `CableConfig::PgMQ`      | `cable_pgmq`   | Postgres + the `pgmq` extension. One ephemeral queue per subscriber.|

The InMem backend is always compiled in (it has no external deps) so unit
tests work without any feature flag. The other four are in `default`.

## Configuration

Add a `cable` section to your environment yaml. The `kind` field selects the
backend:

```yaml
# config/development.yaml — InMem (default for dev)
cable:
  kind: InMem
```

```yaml
# config/production.yaml — Postgres polling
cable:
  kind: Postgres
  uri: {{ get_env(name="DATABASE_URL", default="postgres://...") }}
  polling_interval_ms: 100   # Solid Cable's default
  retention_minutes: 60      # delivered rows are GC'd after this
```

```yaml
# config/production.yaml — Redis pub/sub
cable:
  kind: Redis
  uri: {{ get_env(name="REDIS_URL", default="redis://127.0.0.1:6379") }}
```

```yaml
# config/production.yaml — pgmq
cable:
  kind: PgMQ
  uri: postgres://...
  visibility_timeout_sec: 30
  batch_size: 10
  polling_interval_ms: 100
```

Leave the section out and `AppContext.cable` is `None` — channel routes will
return a clear error when hit.

## Quick start

### 1. Generate a channel

```sh
$ cargo loco generate channel chat
```

This produces `src/channels/chat.rs` with a `Channel` impl, registers it via
`Hooks::register_channels`, and adds the `pub mod chat` entry in
`src/channels/mod.rs`.

```rust
// src/channels/chat.rs (generated)
use loco_rs::prelude::*;

#[derive(Default)]
pub struct Chat;

#[async_trait]
impl Channel for Chat {
    type Params = serde_json::Value;

    async fn subscribed(&self, _ctx: &AppContext, _params: Self::Params) -> Result<Vec<String>> {
        Ok(vec!["chat".to_string()])
    }

    async fn received(&self, _ctx: &AppContext, _data: bytes::Bytes) -> Result<()> {
        Ok(())
    }

    async fn unsubscribed(&self, _ctx: &AppContext) -> Result<()> {
        Ok(())
    }
}
```

### 2. Mount routes

In your controllers, expose the WebSocket and (optionally) SSE endpoints:

```rust
// src/controllers/cable.rs
use loco_rs::{cable::transport, prelude::*};
use crate::channels::chat::Chat;

pub fn routes() -> Routes {
    Routes::new()
        .add("/cable/chat",     get(transport::ws_handler::<Chat>))
        .add("/cable/chat/sse", get(transport::sse_handler::<Chat>))
}
```

Connection-time params are passed through the `?params=...` query string as
JSON and decoded into `Channel::Params`:

```
ws://localhost:5150/cable/chat?params=%7B%22room%22%3A%22rust%22%7D
```

### 3. Publish from anywhere

`AppContext.cable` is an `Option<Cable>` available to every controller, worker
and task:

```rust
async fn broadcast(State(ctx): State<AppContext>) -> Result<Response> {
    if let Some(cable) = &ctx.cable {
        cable
            .publish_json("chat", &serde_json::json!({ "msg": "hello" }))
            .await?;
    }
    format::empty()
}
```

You can also use `cable.publish(topic, bytes)` for raw byte payloads.

## The `Channel` trait

`Channel` shapes one logical group of subscribers. Implementors decide which
topics a connection streams from based on connection-time params, and
optionally handle inbound client messages.

```rust
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    type Params: DeserializeOwned + Send + Sync;

    async fn subscribed(&self, ctx: &AppContext, params: Self::Params)
        -> Result<Vec<String>>;
    async fn received(&self, ctx: &AppContext, data: Bytes) -> Result<()> { Ok(()) }
    async fn unsubscribed(&self, ctx: &AppContext) -> Result<()> { Ok(()) }
}
```

* `subscribed` — return the topics this connection should receive. Common
  pattern: scope by params, e.g. `format!("chat:{}", params.room)`.
* `received` — handle a frame from the client (WebSocket only). Default
  drops it.
* `unsubscribed` — cleanup when the connection closes. Default no-op.

## Low-level `PubSub` primitive

Skipping the `Channel` abstraction, you can program directly against
[`Cable`]:

```rust
let cable = ctx.cable.as_ref().expect("cable configured");

// Producer: publish bytes to a topic.
cable.publish("alerts", Bytes::from("page!")).await?;

// Consumer: subscribe and drain the receiver yourself.
let mut sub = cable.subscribe("alerts").await?;
while let Some(payload) = sub.recv().await {
    tracing::info!(?payload, "got alert");
}
```

The returned `Subscription` cleans up on drop — backends release their
broadcast slot, polling task, or ephemeral queue automatically.

## Backend trade-offs

* **InMem** — fastest, zero overhead. Doesn't survive restarts and doesn't
  fan out across processes. Ideal for tests and single-node development.
* **Postgres polling** — the default Solid-Cable shape: persistent, multi-node,
  ~`polling_interval_ms` worst-case latency. Tune `retention_minutes` to
  control table growth.
* **SQLite polling** — same model as Postgres but local-disk. Great for
  embedded / single-host deployments that still want persistence.
* **Redis** — lowest latency, no DB writes per publish, scales to high fan-out.
  Messages are not persisted, so subscribers only see what is published while
  connected.
* **pgmq** — durable + visibility-timeout semantics give at-least-once delivery
  per subscriber. The cost is one queue per live connection — fine for
  thousands per node, not for massive fan-out (use Redis instead).

## End-to-end testing

Loco ships four runnable examples under `tests/cable/` that boot a real Axum
server, connect a WebSocket client, publish via `ctx.cable`, and assert
delivery — one per non-InMem backend, plus an SSE smoke test for InMem:

```sh
$ cargo test --test mod cable
```

The Postgres / Redis tests connect to `LOCO_TEST_PG_URL` /
`LOCO_TEST_REDIS_URL` (defaulting to a local instance) and skip cleanly if
nothing is reachable.
