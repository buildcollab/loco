//! Cable example #2 — SQLite polling backend.
//!
//! Persists messages to a temp-file SQLite database (`loco_cable_messages`
//! table) and polls every `polling_interval_ms`. Survives a restart, but is
//! single-host (file-system bound).

#![cfg(feature = "cable_sqlt")]

use std::time::Duration;

use bytes::Bytes;
use loco_rs::{
    cable::{sqlt, Cable},
    config::SqliteCableConfig,
};

use super::common;

fn config(path: &str) -> SqliteCableConfig {
    SqliteCableConfig {
        uri: format!("sqlite://{path}?mode=rwc"),
        polling_interval_ms: 50,
        retention_minutes: 60,
        dangerously_flush: true,
    }
}

#[tokio::test]
async fn sqlt_publish_to_websocket_subscriber() {
    let dir = tree_fs::TreeBuilder::default()
        .drop(true)
        .create()
        .expect("tempdir");
    let path = dir.root.join("cable.sqlite");
    let path_str = path.to_str().unwrap().to_string();

    let provider = sqlt::create_provider(&config(&path_str)).await.unwrap();
    let cable = Cable::from_arc(provider);
    let (ctx, router) = common::build_test_app(cable).await;

    let (listener, addr) = common::bind_random_port().await;
    common::serve(listener, router);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let recv_task = tokio::spawn(common::ws_connect_and_recv_one(
        addr,
        "/cable/chat",
        // SQLite poll-interval is 50ms in this test; allow a couple of polls.
        Duration::from_secs(3),
    ));
    // Give the WS upgrade + sub registration time to settle so the polling
    // loop doesn't miss the freshly-inserted row.
    tokio::time::sleep(Duration::from_millis(200)).await;

    ctx.cable
        .as_ref()
        .unwrap()
        .publish("chat:lobby", Bytes::from_static(b"persisted hello"))
        .await
        .unwrap();

    let payload = recv_task.await.expect("ws task");
    assert_eq!(payload, b"persisted hello");
}
