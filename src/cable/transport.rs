//! Axum WebSocket and SSE handlers that bridge a [`super::Cable`] to clients.
//!
//! These handlers are framework-side primitives — application code wires them
//! into routes via the helpers below or directly with Axum:
//!
//! ```rust,ignore
//! use loco_rs::cable::transport;
//! use axum::routing::get;
//!
//! let router = AppRoutes::with_default_routes()
//!     .add_routes(vec![Routes::new()
//!         .add(
//!             "/cable/chat",
//!             get(transport::ws_handler::<MyChatChannel>),
//!         )
//!         .add(
//!             "/cable/chat/sse",
//!             get(transport::sse_handler::<MyChatChannel>),
//!         )]);
//! ```

use std::{convert::Infallible, time::Duration};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use bytes::Bytes;
use futures_util::{stream::Stream, SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{app::AppContext, cable::Channel, Error, Result};

#[derive(Debug, Deserialize)]
pub struct ParamsQuery {
    /// Optional JSON-encoded params blob (`?params={...}`). When absent the
    /// channel receives `serde_json::Value::Null`, which deserializes into
    /// any `Option<T>` or unit-like params type cleanly.
    #[serde(default)]
    pub params: Option<String>,
}

fn parse_params(q: &ParamsQuery) -> Result<serde_json::Value> {
    match &q.params {
        // An empty object deserializes cleanly into any struct whose fields
        // all default; `Null` would only work for `Option<...>` newtypes,
        // which is more surprising.
        None => Ok(serde_json::Value::Object(serde_json::Map::new())),
        Some(raw) => serde_json::from_str(raw).map_err(Error::JSON),
    }
}

/// Axum handler that upgrades a request to a WebSocket and runs `C` against it.
///
/// Mount with `axum::routing::get(ws_handler::<C>)`.
pub async fn ws_handler<C: Channel + Default>(
    ws: WebSocketUpgrade,
    State(ctx): State<AppContext>,
    Query(q): Query<ParamsQuery>,
) -> impl IntoResponse {
    let params = match parse_params(&q) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(error = %err, "cable WS: invalid params");
            return axum::http::StatusCode::BAD_REQUEST.into_response();
        }
    };
    let channel = C::default();
    ws.on_upgrade(move |socket| async move {
        if let Err(err) = run_ws(channel, socket, ctx, params).await {
            tracing::warn!(error = %err, "cable WS terminated with error");
        }
    })
}

async fn run_ws<C: Channel>(
    channel: C,
    socket: WebSocket,
    ctx: AppContext,
    params: serde_json::Value,
) -> Result<()> {
    let cable = ctx
        .cable
        .clone()
        .ok_or_else(|| Error::string("cable provider not configured"))?;

    let topics: Vec<String> = {
        let params_typed: C::Params = serde_json::from_value(params).map_err(Error::JSON)?;
        channel.subscribed(&ctx, params_typed).await?
    };

    let mut subs = Vec::with_capacity(topics.len());
    for topic in &topics {
        subs.push(cable.subscribe(topic).await?);
    }

    let cancel = CancellationToken::new();
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Outbound: drain every subscription into the websocket.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Bytes>();
    for mut sub in subs {
        let tx = out_tx.clone();
        let token = cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = token.cancelled() => break,
                    next = sub.recv() => match next {
                        Some(payload) => {
                            if tx.send(payload).is_err() { break; }
                        }
                        None => break,
                    }
                }
            }
        });
    }
    drop(out_tx); // sub-tasks hold their own clones

    let writer_token = cancel.clone();
    let writer = tokio::spawn(async move {
        loop {
            tokio::select! {
                () = writer_token.cancelled() => break,
                next = out_rx.recv() => match next {
                    Some(payload) => {
                        if ws_tx.send(Message::Binary(payload)).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
        let _ = ws_tx.close().await;
    });

    // Inbound: forward client frames to channel.received.
    let reader_token = cancel.clone();
    let ctx_for_reader = ctx.clone();
    // Move the channel into the reader task so we can call `received` on it
    // and run `unsubscribed` on disconnect.
    let reader = tokio::spawn(async move {
        while let Some(frame) = tokio::select! {
            () = reader_token.cancelled() => None,
            msg = ws_rx.next() => msg,
        } {
            match frame {
                Ok(Message::Text(t)) => {
                    if let Err(err) = channel
                        .received(&ctx_for_reader, Bytes::from(t.to_string()))
                        .await
                    {
                        tracing::warn!(error = %err, "channel.received failed");
                    }
                }
                Ok(Message::Binary(b)) => {
                    if let Err(err) = channel.received(&ctx_for_reader, b).await {
                        tracing::warn!(error = %err, "channel.received failed");
                    }
                }
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(_) => {}
            }
        }
        let _ = channel.unsubscribed(&ctx_for_reader).await;
        reader_token.cancel();
    });

    let _ = tokio::join!(writer, reader);
    cancel.cancel();
    Ok(())
}

/// Axum handler that opens an SSE stream for `C` (server-to-client only).
pub async fn sse_handler<C: Channel + Default>(
    State(ctx): State<AppContext>,
    Query(q): Query<ParamsQuery>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let params = parse_params(&q)?;
    let channel = C::default();

    let cable = ctx
        .cable
        .clone()
        .ok_or_else(|| Error::string("cable provider not configured"))?;

    let topics: Vec<String> = {
        let params_typed: C::Params = serde_json::from_value(params).map_err(Error::JSON)?;
        channel.subscribed(&ctx, params_typed).await?
    };

    let (tx, rx) = mpsc::unbounded_channel::<Event>();

    for topic in topics {
        let mut sub = cable.subscribe(&topic).await?;
        let tx = tx.clone();
        let topic_owned = topic;
        tokio::spawn(async move {
            while let Some(payload) = sub.recv().await {
                // SSE is text-only. UTF-8 payloads pass through unchanged;
                // non-UTF-8 bytes are replaced with U+FFFD via from_utf8_lossy
                // so the stream stays valid. Producers that need lossless
                // binary should use the WS handler.
                let data = String::from_utf8_lossy(&payload).into_owned();
                let event = Event::default().event(topic_owned.clone()).data(data);
                if tx.send(event).is_err() {
                    break;
                }
            }
        });
    }

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|ev| (Ok(ev), rx))
    });
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ka"),
    ))
}
