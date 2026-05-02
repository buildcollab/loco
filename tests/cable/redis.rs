//! Cable example #4 — Redis pub/sub backend.
//!
//! Spins up a real Redis instance via testcontainers, exercises the native
//! `SUBSCRIBE` / `PUBLISH` path. Multi-node out of the box: every Loco
//! process pointed at the same Redis server sees every publish.

#![cfg(feature = "cable_redis")]

use std::time::Duration;

use bytes::Bytes;
use loco_rs::{
    cable::{redis, Cable},
    config::RedisCableConfig,
};

use super::common::{self, redis_url_or_skip};

#[tokio::test]
async fn redis_publish_to_websocket_subscriber() {
    let Some(uri) = redis_url_or_skip().await else {
        return;
    };

    let provider = redis::create_provider(&RedisCableConfig { uri })
        .await
        .expect("create redis cable provider");
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
    // Redis SUBSCRIBE confirmation is async — give the subscription a beat
    // before publishing or the message is dropped.
    tokio::time::sleep(Duration::from_millis(300)).await;

    ctx.cable
        .as_ref()
        .unwrap()
        .publish("chat:lobby", Bytes::from_static(b"hello redis"))
        .await
        .unwrap();

    let payload = recv_task.await.expect("ws task");
    assert_eq!(payload, b"hello redis");
}
