//! Cable example #3 — Postgres polling backend.
//!
//! Spins up a real Postgres instance via testcontainers, publishes a row to
//! `loco_cable_messages`, and verifies a WS client subscribed through Loco
//! receives it.

#![cfg(feature = "cable_pg")]

use std::time::Duration;

use bytes::Bytes;
use loco_rs::{
    cable::{pg, Cable},
    config::PostgresCableConfig,
};

use super::common::{self, pg_url_or_skip};

#[tokio::test]
async fn pg_publish_to_websocket_subscriber() {
    let Some(uri) = pg_url_or_skip().await else {
        return;
    };

    let provider = pg::create_provider(&PostgresCableConfig {
        uri: uri.clone(),
        polling_interval_ms: 50,
        retention_minutes: 60,
        dangerously_flush: true,
    })
    .await
    .expect("create pg cable provider");

    let cable = Cable::from_arc(provider);
    let (ctx, router) = common::build_test_app(cable).await;

    let (listener, addr) = common::bind_random_port().await;
    common::serve(listener, router);
    tokio::time::sleep(Duration::from_millis(100)).await;

    let recv_task = tokio::spawn(common::ws_connect_and_recv_one(
        addr,
        "/cable/chat",
        Duration::from_secs(5),
    ));
    tokio::time::sleep(Duration::from_millis(250)).await;

    ctx.cable
        .as_ref()
        .unwrap()
        .publish("chat:lobby", Bytes::from_static(b"hello pg"))
        .await
        .unwrap();

    let payload = recv_task.await.expect("ws task");
    assert_eq!(payload, b"hello pg");
}
