//! Cable integration tests — Cable ↔ Channel ↔ ChannelRegistry.
#![cfg(test)]

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;

use crate::{
    app::AppContext,
    cable::{inmem::InMemPubSub, Cable, Channel, ChannelRegistry},
    tests_cfg, Result,
};

#[derive(Debug, Deserialize, Default)]
struct ChatParams {
    #[serde(default)]
    room: Option<String>,
}

#[derive(Default)]
struct ChatChannel;

#[async_trait]
impl Channel for ChatChannel {
    type Params = ChatParams;

    async fn subscribed(&self, _ctx: &AppContext, params: Self::Params) -> Result<Vec<String>> {
        let room = params.room.unwrap_or_else(|| "lobby".to_string());
        Ok(vec![format!("chat:{room}")])
    }
}

#[tokio::test]
async fn cable_wraps_inmem_provider() {
    let cable = Cable::new(InMemPubSub::default());

    let mut sub = cable.subscribe("t").await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;

    cable
        .publish_json("t", &serde_json::json!({"hello": "world"}))
        .await
        .unwrap();

    let payload = tokio::time::timeout(Duration::from_secs(1), sub.recv())
        .await
        .unwrap()
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(parsed["hello"], "world");
}

#[tokio::test]
async fn channel_registry_resolves_by_name() {
    let mut reg = ChannelRegistry::new();
    reg.register("chat", ChatChannel);
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    let dyn_channel = reg.get("chat").expect("chat channel registered");

    let mut ctx = tests_cfg::app::get_app_context().await;
    ctx.cable = Some(Cable::new(InMemPubSub::default()));

    let topics = dyn_channel
        .subscribed_dyn(&ctx, serde_json::json!({"room": "rust"}))
        .await
        .unwrap();
    assert_eq!(topics, vec!["chat:rust".to_string()]);

    // Empty-object params should fall back to the lobby topic.
    let topics_default = dyn_channel
        .subscribed_dyn(&ctx, serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(topics_default, vec!["chat:lobby".to_string()]);
}

#[tokio::test]
async fn end_to_end_publish_via_appcontext() {
    let mut ctx = tests_cfg::app::get_app_context().await;
    let cable = Cable::new(InMemPubSub::default());
    ctx.cable = Some(cable.clone());

    let channel = ChatChannel;
    let topics = channel
        .subscribed(
            &ctx,
            ChatParams {
                room: Some("dev".into()),
            },
        )
        .await
        .unwrap();
    assert_eq!(topics, vec!["chat:dev".to_string()]);

    let mut sub = cable.subscribe(&topics[0]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    cable
        .publish(&topics[0], Bytes::from_static(b"hi"))
        .await
        .unwrap();
    let payload = tokio::time::timeout(Duration::from_secs(1), sub.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(payload.as_ref(), b"hi");
}
