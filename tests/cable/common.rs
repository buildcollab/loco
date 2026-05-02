//! Shared scaffolding for the four cable end-to-end "example" tests.
//!
//! Each backend test wires the same `ChatChannel`, the same routes, and the
//! same WS-client / SSE-client logic — only the [`loco_rs::cable::Cable`]
//! provider differs. The shared bits live here.

use std::{net::SocketAddr, sync::OnceLock, time::Duration};

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use loco_rs::{
    cable::{transport, Cable, Channel},
    controller::AppRoutes,
    prelude::*,
};
use serde::Deserialize;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Bind to an ephemeral port so multiple example tests can run in parallel
/// without colliding on a fixed port.
pub async fn bind_random_port() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    (listener, addr)
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatParams {
    #[serde(default)]
    pub room: Option<String>,
}

#[derive(Default)]
pub struct ChatChannel;

#[async_trait]
impl Channel for ChatChannel {
    type Params = ChatParams;

    async fn subscribed(&self, _ctx: &AppContext, params: Self::Params) -> Result<Vec<String>> {
        let room = params.room.unwrap_or_else(|| "lobby".to_string());
        Ok(vec![format!("chat:{room}")])
    }
}

/// Build a minimal `AppContext` wired with the supplied [`Cable`] provider
/// and a router that exposes `/cable/chat` (WS) + `/cable/chat/sse` (SSE).
pub async fn build_test_app(cable: Cable) -> (AppContext, axum::Router) {
    let mut ctx = loco_rs::tests_cfg::app::get_app_context().await;
    ctx.cable = Some(cable);

    let routes = AppRoutes::empty()
        .add_route(
            Routes::new()
                .add(
                    "/cable/chat",
                    get(transport::ws_handler::<ChatChannel>),
                )
                .add(
                    "/cable/chat/sse",
                    get(transport::sse_handler::<ChatChannel>),
                ),
        )
        .to_router::<loco_rs::tests_cfg::db::AppHook>(ctx.clone(), axum::Router::new())
        .expect("build router");

    (ctx, routes)
}

/// Boot a real Axum server on `addr` in a spawned task. The returned handle
/// is forgotten — the OS reaps it when the test exits.
pub fn serve(listener: TcpListener, router: axum::Router) {
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    // Give the listener a tick to start polling.
    static WAIT: OnceLock<()> = OnceLock::new();
    WAIT.get_or_init(|| {});
}

/// Connect a WebSocket client and drain until we either receive `predicate`
/// returning `Some(value)` or hit the timeout.
pub async fn ws_connect_and_recv_one(
    addr: SocketAddr,
    path: &str,
    timeout: Duration,
) -> Vec<u8> {
    let url = format!("ws://{addr}{path}");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws handshake");

    let payload = tokio::time::timeout(timeout, async {
        loop {
            match ws.next().await {
                Some(Ok(WsMessage::Binary(b))) => return b.to_vec(),
                Some(Ok(WsMessage::Text(t))) => return t.as_bytes().to_vec(),
                Some(Ok(WsMessage::Ping(_))) => {
                    let _ = ws.send(WsMessage::Pong(Vec::new().into())).await;
                }
                Some(Ok(_)) => {}
                Some(Err(err)) => panic!("ws read error: {err}"),
                None => panic!("ws closed before message"),
            }
        }
    })
    .await
    .expect("ws timed out waiting for first message");

    let _ = ws.close(None).await;
    payload
}

/// Resolve a Postgres URL: env override `LOCO_TEST_PG_URL`, then default to
/// a local instance. Returns `None` (and the test should skip) when nothing
/// is reachable.
#[cfg(feature = "cable_pg")]
pub async fn pg_url_or_skip() -> Option<String> {
    use sqlx::PgPool;

    let uri = std::env::var("LOCO_TEST_PG_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@127.0.0.1:5432/loco_test".to_string());

    for _ in 0..3 {
        if let Ok(pool) = PgPool::connect(&uri).await {
            if sqlx::query("SELECT 1").execute(&pool).await.is_ok() {
                return Some(uri);
            }
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    eprintln!(
        "[cable::pg] no Postgres reachable at {uri} — skipping. Set LOCO_TEST_PG_URL to override."
    );
    None
}

/// Resolve a Redis URL: env override `LOCO_TEST_REDIS_URL`, then default to
/// a local instance. Returns `None` (and the test should skip) when nothing
/// is reachable.
#[cfg(feature = "cable_redis")]
pub async fn redis_url_or_skip() -> Option<String> {
    use redis::Client;

    let uri =
        std::env::var("LOCO_TEST_REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    if let Ok(client) = Client::open(uri.clone()) {
        for _ in 0..3 {
            if let Ok(mut conn) = client.get_multiplexed_async_connection().await {
                if redis::cmd("PING")
                    .query_async::<()>(&mut conn)
                    .await
                    .is_ok()
                {
                    return Some(uri);
                }
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
    eprintln!(
        "[cable::redis] no Redis reachable at {uri} — skipping. Set LOCO_TEST_REDIS_URL to override."
    );
    None
}

/// Subscribe to the SSE endpoint and return the first `data:` line we see.
pub async fn sse_connect_and_recv_one(
    addr: SocketAddr,
    path: &str,
    timeout: Duration,
) -> String {
    let url = format!("http://{addr}{path}");
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .expect("sse get");
    assert!(resp.status().is_success(), "sse status: {}", resp.status());

    let mut stream = resp.bytes_stream();
    tokio::time::timeout(timeout, async {
        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk: bytes::Bytes = chunk.expect("sse chunk");
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // SSE messages end with `\n\n`. Find the first event with a
            // `data:` line and return its payload.
            while let Some(idx) = buf.find("\n\n") {
                let frame: String = buf.drain(..=idx + 1).collect();
                for line in frame.lines() {
                    if let Some(rest) = line.strip_prefix("data:") {
                        return rest.trim().to_string();
                    }
                }
            }
        }
        panic!("sse closed before data");
    })
    .await
    .expect("sse timeout")
}
