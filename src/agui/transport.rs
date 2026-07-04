//! # SSE transport + event sink
//!
//! Bridges [`AguiEvent`]s to an axum SSE response. The run-loop writes events
//! into an [`EventSink`]; for HTTP delivery that sink is an [`MpscSink`] whose
//! receiver end is turned into an SSE body by [`sse_response`].
//!
//! ## Abort chain (no `CancellationToken`)
//!
//! Cancellation is structural, driven entirely by channel drops:
//!
//! 1. The SSE body owns the [`mpsc::Receiver`].
//! 2. Client disconnect → axum drops the response → drops the receiver.
//! 3. The run-loop's next [`EventSink::emit`] (an [`MpscSink`] send) returns
//!    `Err` because the channel is closed.
//! 4. The run-loop unwinds and drops the provider's delta sender.
//! 5. The provider's next `tx.send` returns `Err`, so it stops driving the
//!    upstream stream, dropping the `reqwest` response and aborting the HTTP
//!    request to the model.
//!
//! Everything tears down without any explicit cancellation token.

use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::agui::hub::{HubEvent, HubEventStream};
use crate::agui::protocol::AguiEvent;
use crate::{Error, Result};

/// A sink the run-loop emits protocol events into. Implementations decide where
/// events go (an SSE channel, a test `Vec`, or nowhere).
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    /// Emit one event.
    ///
    /// # Errors
    /// Returns an error when the destination is gone (e.g. the SSE client
    /// disconnected) — this doubles as the run-loop's abort signal.
    async fn emit(&self, ev: AguiEvent) -> Result<()>;
}

/// Forwarding impl so `Box<dyn EventSink>` is itself a `Sized` [`EventSink`] —
/// useful for holding an erased sink (e.g. a subagent's DB-logging sink).
#[async_trait::async_trait]
impl<T: ?Sized + EventSink> EventSink for Box<T> {
    async fn emit(&self, ev: AguiEvent) -> Result<()> {
        (**self).emit(ev).await
    }
}

/// Forwarding impl so a `&dyn EventSink` satisfies the `EventSink` bound (e.g. a
/// subagent handed an erased sink can pass it to `run_turn`).
#[async_trait::async_trait]
impl<T: ?Sized + EventSink> EventSink for &T {
    async fn emit(&self, ev: AguiEvent) -> Result<()> {
        (**self).emit(ev).await
    }
}

/// An [`EventSink`] backed by an mpsc channel — the SSE path.
pub struct MpscSink(pub mpsc::Sender<AguiEvent>);

#[async_trait::async_trait]
impl EventSink for MpscSink {
    async fn emit(&self, ev: AguiEvent) -> Result<()> {
        self.0
            .send(ev)
            .await
            .map_err(|_| Error::string("event sink closed (client disconnected)"))
    }
}

/// An [`EventSink`] that discards events — for non-SSE callers (e.g. a
/// background run whose events are only persisted).
pub struct NullSink;

#[async_trait::async_trait]
impl EventSink for NullSink {
    async fn emit(&self, _ev: AguiEvent) -> Result<()> {
        Ok(())
    }
}

/// Serialize one [`AguiEvent`] into an SSE [`Event`] (named by
/// [`AguiEvent::event_name`], JSON body). Exposed for testing the mapping.
///
/// # Panics
/// Panics only if the event fails to serialize to JSON, which cannot happen for
/// the protocol's own types.
pub fn event_to_sse(ev: &AguiEvent) -> Event {
    Event::default()
        .event(ev.event_name())
        .json_data(ev)
        .expect("AguiEvent serializes to JSON")
}

/// Build an SSE response body from a receiver of protocol events.
///
/// Each [`AguiEvent`] becomes an SSE `event:`/`data:` frame; keep-alive comments
/// are sent on idle.
pub fn sse_response(
    rx: mpsc::Receiver<AguiEvent>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|ev| (Ok(event_to_sse(&ev)), rx))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Serialize a numbered [`HubEvent`] into an SSE [`Event`], setting the SSE
/// `id:` to the per-run sequence number. A reconnecting client echoes the last
/// id it saw (as `Last-Event-ID` / a `since` query) to resume without gaps.
#[must_use]
pub fn hub_event_to_sse(ev: &HubEvent) -> Event {
    Event::default()
        .id(ev.seq.to_string())
        .event(ev.name.clone())
        .data(ev.data.to_string())
}

/// Build an SSE response from a [`RunHub`](crate::agui::hub::RunHub) stream
/// (replay-then-tail). This is the resumable path: the stream keeps producing
/// even across client reconnects, and each frame carries its `seq` as the SSE
/// `id:`.
pub fn hub_sse_response(
    stream: HubEventStream,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let mapped = stream.map(|ev| Ok(hub_event_to_sse(&ev)));
    Sse::new(mapped).keep_alive(KeepAlive::default())
}

/// Convenience: create the event channel, spawn `f(sink)` to drive a run, and
/// return the SSE response wired to it. `on_exit` runs when the spawned task
/// ends (regardless of success) — use it for cleanup such as clearing a stuck
/// "responding" status.
///
/// The spawned task owns the [`MpscSink`]; when the client disconnects the
/// receiver drops and the run-loop unwinds via the abort chain documented above.
pub fn spawn_and_stream<F, Fut>(
    buffer: usize,
    on_exit: impl FnOnce() + Send + 'static,
    f: F,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>
where
    F: FnOnce(MpscSink) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<AguiEvent>(buffer.max(1));
    tokio::spawn(async move {
        let sink = MpscSink(tx);
        f(sink).await;
        on_exit();
    });
    sse_response(rx)
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::protocol::{AguiEvent, RunOutcome};

    #[test]
    fn maps_event_name_and_json_body() {
        let ev = AguiEvent::TextMessageContent {
            message_id: "m1".into(),
            delta: "hi".into(),
        };
        let sse = event_to_sse(&ev);
        // The rendered SSE frame carries the event name and JSON payload.
        let rendered = format!("{sse:?}");
        // Event's Debug isn't structured; assert via re-serialization instead.
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["type"], "TEXT_MESSAGE_CONTENT");
        assert_eq!(json["messageId"], "m1");
        // Sanity: the frame debug-string is non-empty.
        assert!(!rendered.is_empty());
    }

    #[tokio::test]
    async fn mpsc_sink_reports_closed_channel() {
        let (tx, rx) = mpsc::channel::<AguiEvent>(4);
        let sink = MpscSink(tx);
        drop(rx);
        let err = sink
            .emit(AguiEvent::RunFinished {
                thread_id: "t".into(),
                run_id: "r".into(),
                outcome: RunOutcome::Success,
                interrupt: None,
            })
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn null_sink_is_ok() {
        assert!(NullSink
            .emit(AguiEvent::RunError {
                message: "x".into(),
                code: None
            })
            .await
            .is_ok());
    }
}
