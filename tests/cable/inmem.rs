//! Cable example #1 — InMem backend (single-process, always available).
//!
//! Demonstrates the simplest setup: tokio broadcast under the hood, no
//! external services. A WS client connects, the server publishes via
//! `ctx.cable`, the client receives.

use std::time::Duration;

use bytes::Bytes;
use loco_rs::cable::{inmem::InMemPubSub, Cable};

use super::common;

#[tokio::test]
async fn inmem_publish_to_websocket_subscriber() {
    let cable = Cable::new(InMemPubSub::default());
    let (ctx, router) = common::build_test_app(cable.clone()).await;

    let (listener, addr) = common::bind_random_port().await;
    common::serve(listener, router);
    // Tiny grace period for the listener task.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Client subscribes to the lobby.
    let recv_task = tokio::spawn(common::ws_connect_and_recv_one(
        addr,
        "/cable/chat",
        Duration::from_secs(2),
    ));

    // Give the WS upgrade + Channel::subscribed a moment to install.
    tokio::time::sleep(Duration::from_millis(100)).await;

    ctx.cable
        .as_ref()
        .unwrap()
        .publish("chat:lobby", Bytes::from_static(b"hello from inmem"))
        .await
        .unwrap();

    let payload = recv_task.await.expect("ws task");
    assert_eq!(payload, b"hello from inmem");
}

#[tokio::test]
async fn inmem_publish_to_sse_subscriber() {
    let cable = Cable::new(InMemPubSub::default());
    let (ctx, router) = common::build_test_app(cable).await;

    let (listener, addr) = common::bind_random_port().await;
    common::serve(listener, router);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let recv_task = tokio::spawn(common::sse_connect_and_recv_one(
        addr,
        "/cable/chat/sse?params=%7B%22room%22%3A%22sse%22%7D",
        Duration::from_secs(2),
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;

    ctx.cable
        .as_ref()
        .unwrap()
        .publish("chat:sse", Bytes::from_static(b"hello sse"))
        .await
        .unwrap();

    let payload = recv_task.await.expect("sse task");
    assert_eq!(payload, "hello sse");
}
