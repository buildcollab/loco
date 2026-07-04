//! # Run hub — resumable + cancellable streaming
//!
//! The [`RunHub`] is the backbone that **decouples a run from the HTTP
//! connection** so streams survive network glitches and a run can be explicitly
//! cancelled. It replaces the "structural cancellation via channel drop" model:
//!
//! - The run task writes events into the hub through a [`HubSink`] (never fails
//!   on client disconnect — the run keeps producing).
//! - An HTTP handler [`subscribe`](RunHub::subscribe)s from a sequence number,
//!   replaying buffered events then tailing live ones — so a reconnecting client
//!   resumes exactly where it left off (`GET .../stream?since=N`).
//! - Every event carries a monotonic per-run `seq` (surfaced as the SSE `id:`)
//!   so the client knows what to resume from.
//! - A [`RunHandle::cancel`] token, flipped by [`cancel`](RunHub::cancel)
//!   (`POST .../cancel`), lets a "stop" request halt the run cooperatively.
//!
//! ## Backends
//!
//! - [`InMemoryRunHub`] (here) — a per-process buffer + broadcast. Single node.
//! - [`DbRunHub`] (here, behind `with-db`) — a multi-node backend that persists
//!   events to `agent_events` and coordinates cancellation via `agent_runs`,
//!   over the framework-owned [`entities`](super::entities). [`run_hub`] picks
//!   between them from the `agui.hub` config.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::agui::protocol::AguiEvent;
use crate::agui::transport::EventSink;
use crate::Result;

/// Default number of events retained per run for replay-on-reconnect.
pub const DEFAULT_BUFFER_CAP: usize = 1024;

/// A pre-serialized protocol event with its per-run sequence number. Using a
/// serialized form (name + JSON) lets every backend — in-memory or DB — carry
/// events uniformly without round-tripping through [`AguiEvent`]
/// (which is serialize-only).
#[derive(Debug, Clone)]
pub struct HubEvent {
    /// Monotonic per-run sequence number (starts at 1). The SSE `id:`.
    pub seq: u64,
    /// The AG-UI event name (SSE `event:`), e.g. `TEXT_MESSAGE_CONTENT`.
    pub name: String,
    /// The event payload as JSON (SSE `data:`).
    pub data: Value,
}

impl HubEvent {
    /// Serialize an [`AguiEvent`] into a numbered hub event.
    #[must_use]
    pub fn from_event(seq: u64, ev: &AguiEvent) -> Self {
        Self {
            seq,
            name: ev.event_name().to_string(),
            data: serde_json::to_value(ev).unwrap_or(Value::Null),
        }
    }
}

/// A handle to a started run: its id and a cooperative cancellation token the
/// run-loop polls.
#[derive(Debug, Clone)]
pub struct RunHandle {
    /// The run id.
    pub run_id: String,
    /// Cancellation token — flipped by [`RunHub::cancel`] / a stop request.
    pub cancel: CancellationToken,
}

/// A stream of numbered events for a run (replay-then-tail).
pub type HubEventStream = Pin<Box<dyn Stream<Item = HubEvent> + Send>>;

/// The seam that decouples a run from its client connection. Object-safe so it
/// can be held as `Arc<dyn RunHub>`.
#[async_trait]
pub trait RunHub: Send + Sync {
    /// Begin a run: allocate its buffer + cancellation token. Returns the handle
    /// whose token the run-loop must poll.
    ///
    /// # Errors
    /// Backend errors (e.g. persisting the run record) propagate.
    async fn start(&self, run_id: &str) -> Result<RunHandle>;

    /// Publish an event to a run (assigns the next `seq`, buffers, fans out).
    ///
    /// # Errors
    /// Backend errors propagate.
    async fn publish(&self, run_id: &str, ev: &AguiEvent) -> Result<()>;

    /// Subscribe to a run from just after `since` (0 = from the start): replay
    /// buffered events with `seq > since`, then tail live ones.
    ///
    /// # Errors
    /// Backend errors propagate.
    async fn subscribe(&self, run_id: &str, since: u64) -> Result<HubEventStream>;

    /// Request cancellation of a run. Returns whether a matching active run was
    /// found. The run-loop notices via its [`RunHandle::cancel`] token.
    ///
    /// # Errors
    /// Backend errors propagate.
    async fn cancel(&self, run_id: &str) -> Result<bool>;

    /// Mark a run finished. Backends may retain the buffer briefly so
    /// late reconnects can still replay the final events, then GC it.
    ///
    /// # Errors
    /// Backend errors propagate.
    async fn finish(&self, run_id: &str) -> Result<()>;
}

/// Forwarding impl so `Arc<dyn RunHub>` is itself a [`RunHub`].
#[async_trait]
impl<T: ?Sized + RunHub> RunHub for Arc<T> {
    async fn start(&self, run_id: &str) -> Result<RunHandle> {
        (**self).start(run_id).await
    }
    async fn publish(&self, run_id: &str, ev: &AguiEvent) -> Result<()> {
        (**self).publish(run_id, ev).await
    }
    async fn subscribe(&self, run_id: &str, since: u64) -> Result<HubEventStream> {
        (**self).subscribe(run_id, since).await
    }
    async fn cancel(&self, run_id: &str) -> Result<bool> {
        (**self).cancel(run_id).await
    }
    async fn finish(&self, run_id: &str) -> Result<()> {
        (**self).finish(run_id).await
    }
}

/// An [`EventSink`] that writes run events into a [`RunHub`].
///
/// Unlike [`MpscSink`](crate::agui::transport::MpscSink), **`emit` never fails
/// on client disconnect** — that is the whole point: the run is decoupled from
/// the connection and only stops via its cancellation token or normal
/// completion.
pub struct HubSink {
    hub: Arc<dyn RunHub>,
    run_id: String,
}

impl HubSink {
    /// Build a sink that publishes to `run_id` on `hub`.
    #[must_use]
    pub fn new(hub: Arc<dyn RunHub>, run_id: impl Into<String>) -> Self {
        Self {
            hub,
            run_id: run_id.into(),
        }
    }
}

#[async_trait]
impl EventSink for HubSink {
    async fn emit(&self, ev: AguiEvent) -> Result<()> {
        // Publish failures (e.g. a transient DB error) are logged but do not
        // abort the run — losing one buffered frame must not kill generation.
        if let Err(e) = self.hub.publish(&self.run_id, &ev).await {
            tracing::warn!(target: "loco_rs::agui", error = %e, "run hub publish failed");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

struct RunState {
    /// Last assigned seq (next is `seq + 1`).
    seq: u64,
    /// Bounded replay buffer (drop-oldest when full).
    buffer: Vec<HubEvent>,
    /// Live fan-out to current subscribers.
    tx: broadcast::Sender<HubEvent>,
    /// Cancellation token for this run.
    cancel: CancellationToken,
}

/// A process-local [`RunHub`]: a bounded per-run buffer plus a broadcast for
/// live tailing. Single-node only (no cross-process visibility).
#[derive(Clone)]
pub struct InMemoryRunHub {
    inner: Arc<Inner>,
}

struct Inner {
    runs: Mutex<HashMap<String, RunState>>,
    buffer_cap: usize,
    /// Live-broadcast channel capacity.
    channel_cap: usize,
}

impl Default for InMemoryRunHub {
    fn default() -> Self {
        Self::new(DEFAULT_BUFFER_CAP)
    }
}

impl InMemoryRunHub {
    /// Create a hub retaining up to `buffer_cap` events per run for replay.
    #[must_use]
    pub fn new(buffer_cap: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                runs: Mutex::new(HashMap::new()),
                buffer_cap: buffer_cap.max(1),
                channel_cap: 256,
            }),
        }
    }
}

#[async_trait]
impl RunHub for InMemoryRunHub {
    async fn start(&self, run_id: &str) -> Result<RunHandle> {
        let (tx, _rx) = broadcast::channel(self.inner.channel_cap);
        let cancel = CancellationToken::new();
        let mut runs = self.inner.runs.lock().expect("run hub mutex");
        runs.insert(
            run_id.to_string(),
            RunState {
                seq: 0,
                buffer: Vec::new(),
                tx,
                cancel: cancel.clone(),
            },
        );
        Ok(RunHandle {
            run_id: run_id.to_string(),
            cancel,
        })
    }

    async fn publish(&self, run_id: &str, ev: &AguiEvent) -> Result<()> {
        let mut runs = self.inner.runs.lock().expect("run hub mutex");
        let Some(state) = runs.get_mut(run_id) else {
            // Run not registered (already finished/GC'd) — drop silently.
            return Ok(());
        };
        state.seq += 1;
        let he = HubEvent::from_event(state.seq, ev);
        state.buffer.push(he.clone());
        let overflow = state.buffer.len().saturating_sub(self.inner.buffer_cap);
        if overflow > 0 {
            state.buffer.drain(0..overflow);
        }
        // Ignore send errors — no live subscribers is fine (replay covers them).
        let _ = state.tx.send(he);
        Ok(())
    }

    async fn subscribe(&self, run_id: &str, since: u64) -> Result<HubEventStream> {
        let (replay, rx) = {
            let runs = self.inner.runs.lock().expect("run hub mutex");
            match runs.get(run_id) {
                Some(state) => {
                    // Subscribe under the lock so no event can slip between the
                    // buffer snapshot and the live subscription.
                    let rx = state.tx.subscribe();
                    let replay: Vec<HubEvent> = state
                        .buffer
                        .iter()
                        .filter(|e| e.seq > since)
                        .cloned()
                        .collect();
                    (replay, Some(rx))
                }
                None => (Vec::new(), None),
            }
        };

        let replay_stream = futures_util::stream::iter(replay);
        let Some(rx) = rx else {
            // Unknown run: just yield whatever (nothing) — caller handles 204.
            return Ok(Box::pin(replay_stream));
        };

        // Live tail: buffer events already covered by `replay` (seq <= snapshot)
        // cannot appear here because publish holds the same lock; still guard
        // with `seq > since` for safety. Lagged receivers skip missed frames.
        let live = futures_util::stream::unfold(rx, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => return Some((ev, rx)),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        })
        .filter(move |e| futures_util::future::ready(e.seq > since));

        Ok(Box::pin(replay_stream.chain(live)))
    }

    async fn cancel(&self, run_id: &str) -> Result<bool> {
        let runs = self.inner.runs.lock().expect("run hub mutex");
        match runs.get(run_id) {
            Some(state) => {
                state.cancel.cancel();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn finish(&self, run_id: &str) -> Result<()> {
        // Retain the buffer for a short grace period so a client mid-reconnect
        // can still replay the terminal events, then GC.
        let inner = self.inner.clone();
        let run_id = run_id.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            inner.runs.lock().expect("run hub mutex").remove(&run_id);
        });
        Ok(())
    }
}

/// Wrap an mpsc receiver of [`HubEvent`]s as a [`HubEventStream`]. The DB-backed
/// [`DbRunHub`] polls its event table into a channel and uses this to expose it
/// as a stream — keeping the stream-combinator glue in one place.
#[must_use]
pub fn channel_stream(rx: tokio::sync::mpsc::Receiver<HubEvent>) -> HubEventStream {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|ev| (ev, rx))
    }))
}

/// Build the framework's in-memory hub as a trait object. For multi-node
/// deploys, [`run_hub`] returns a [`DbRunHub`] instead based on `agui.hub`; this
/// helper only covers the single-process backend.
#[must_use]
pub fn in_memory() -> Arc<dyn RunHub> {
    Arc::new(InMemoryRunHub::default())
}

// ---------------------------------------------------------------------------
// DB-backed backend (multi-node) + hub selection
// ---------------------------------------------------------------------------

#[cfg(feature = "with-db")]
mod db {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;

    use async_trait::async_trait;
    use sea_orm::{
        ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel,
        QueryFilter, QueryOrder, Set,
    };
    use serde_json::Value;
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    use super::{channel_stream, HubEvent, HubEventStream, RunHandle, RunHub};
    use crate::agui::entities::{agent_events, agent_runs};
    use crate::agui::protocol::AguiEvent;
    use crate::app::AppContext;
    use crate::config::HubConfig;
    use crate::Result;

    /// Process-wide run hub, selected once from `agui.hub` config.
    static HUB: OnceLock<Arc<dyn RunHub>> = OnceLock::new();

    /// The process-wide run hub. In-memory for single-node; DB-backed
    /// (multi-node) when `agui.hub` is `redis` or `postgres`.
    ///
    /// This is the single construction point the framework controller and the
    /// durable worker both use, so a given process streams and produces through
    /// the same backend.
    #[must_use]
    pub fn run_hub(ctx: &AppContext) -> Arc<dyn RunHub> {
        HUB.get_or_init(|| {
            let kind = ctx
                .config
                .agui
                .as_ref()
                .map(|a| a.hub.clone())
                .unwrap_or_default();
            match kind {
                HubConfig::InMem => super::in_memory(),
                HubConfig::Redis | HubConfig::Postgres => Arc::new(DbRunHub::new(ctx.db.clone())),
            }
        })
        .clone()
    }

    /// A multi-node [`RunHub`]: events persist to `agent_events` (replayed on
    /// resume), and cancellation rides on `agent_runs.cancel_requested` — polled
    /// by the node that owns the run, which flips its local token. Live tailing
    /// is by polling the shared tables, so any node can serve a reconnect (and a
    /// [worker](crate::agui::worker) on one node can drive a run that a web node
    /// streams).
    pub struct DbRunHub {
        db: DatabaseConnection,
        /// Per-run publish sequence (only the owning node publishes a given run).
        seqs: Mutex<HashMap<String, i64>>,
    }

    impl DbRunHub {
        /// Build a DB-backed hub over `db`.
        #[must_use]
        pub fn new(db: DatabaseConnection) -> Self {
            Self {
                db,
                seqs: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl RunHub for DbRunHub {
        async fn start(&self, run_id: &str) -> Result<RunHandle> {
            let existing = agent_runs::Entity::find()
                .filter(agent_runs::Column::RunId.eq(run_id))
                .one(&self.db)
                .await?;
            if existing.is_none() {
                agent_runs::ActiveModel {
                    pid: Set(Uuid::new_v4()),
                    run_id: Set(run_id.to_string()),
                    status: Set("running".to_string()),
                    cancel_requested: Set(false),
                    last_seq: Set(0),
                    ..Default::default()
                }
                .insert(&self.db)
                .await?;
            }
            self.seqs
                .lock()
                .expect("seqs mutex")
                .insert(run_id.to_string(), 0);

            // Poll for a cross-node cancel request; flip the local token when seen.
            let cancel = CancellationToken::new();
            let poll_token = cancel.clone();
            let db = self.db.clone();
            let rid = run_id.to_string();
            tokio::spawn(async move {
                loop {
                    if poll_token.is_cancelled() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(1000)).await;
                    let row = agent_runs::Entity::find()
                        .filter(agent_runs::Column::RunId.eq(&rid))
                        .one(&db)
                        .await
                        .ok()
                        .flatten();
                    match row {
                        Some(r) if r.cancel_requested => {
                            poll_token.cancel();
                            break;
                        }
                        // The run finished (possibly on another node). Stop
                        // polling so a node that only created the row — e.g. a
                        // web node in worker mode — doesn't leak this task.
                        Some(r)
                            if matches!(
                                r.status.as_str(),
                                "complete" | "errored" | "cancelled"
                            ) =>
                        {
                            break;
                        }
                        _ => {}
                    }
                }
            });

            Ok(RunHandle {
                run_id: run_id.to_string(),
                cancel,
            })
        }

        async fn publish(&self, run_id: &str, ev: &AguiEvent) -> Result<()> {
            let seq = {
                let mut m = self.seqs.lock().expect("seqs mutex");
                let e = m.entry(run_id.to_string()).or_insert(0);
                *e += 1;
                *e
            };
            let he = HubEvent::from_event(seq as u64, ev);
            agent_events::ActiveModel {
                pid: Set(Uuid::new_v4()),
                run_id: Set(run_id.to_string()),
                seq: Set(seq),
                name: Set(he.name.clone()),
                payload: Set(Some(he.data.clone())),
                ..Default::default()
            }
            .insert(&self.db)
            .await?;
            if let Some(row) = agent_runs::Entity::find()
                .filter(agent_runs::Column::RunId.eq(run_id))
                .one(&self.db)
                .await?
            {
                let mut am = row.into_active_model();
                am.last_seq = Set(seq);
                am.update(&self.db).await?;
            }
            Ok(())
        }

        async fn subscribe(&self, run_id: &str, since: u64) -> Result<HubEventStream> {
            let (tx, rx) = tokio::sync::mpsc::channel::<HubEvent>(256);
            let db = self.db.clone();
            let rid = run_id.to_string();
            tokio::spawn(async move {
                let mut last = i64::try_from(since).unwrap_or(0);
                loop {
                    let events = agent_events::Entity::find()
                        .filter(agent_events::Column::RunId.eq(&rid))
                        .filter(agent_events::Column::Seq.gt(last))
                        .order_by_asc(agent_events::Column::Seq)
                        .all(&db)
                        .await
                        .unwrap_or_default();
                    for e in events {
                        last = e.seq;
                        let he = HubEvent {
                            seq: e.seq as u64,
                            name: e.name,
                            data: e.payload.unwrap_or(Value::Null),
                        };
                        if tx.send(he).await.is_err() {
                            return; // client gone
                        }
                    }
                    let run_row = agent_runs::Entity::find()
                        .filter(agent_runs::Column::RunId.eq(&rid))
                        .one(&db)
                        .await
                        .ok()
                        .flatten();
                    let done = match run_row {
                        Some(r) => {
                            matches!(r.status.as_str(), "complete" | "errored" | "cancelled")
                        }
                        None => true,
                    };
                    if done {
                        // Final drain to catch events written just before terminal.
                        let tail = agent_events::Entity::find()
                            .filter(agent_events::Column::RunId.eq(&rid))
                            .filter(agent_events::Column::Seq.gt(last))
                            .order_by_asc(agent_events::Column::Seq)
                            .all(&db)
                            .await
                            .unwrap_or_default();
                        for e in tail {
                            let he = HubEvent {
                                seq: e.seq as u64,
                                name: e.name,
                                data: e.payload.unwrap_or(Value::Null),
                            };
                            let _ = tx.send(he).await;
                        }
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            });
            Ok(channel_stream(rx))
        }

        async fn cancel(&self, run_id: &str) -> Result<bool> {
            let Some(row) = agent_runs::Entity::find()
                .filter(agent_runs::Column::RunId.eq(run_id))
                .one(&self.db)
                .await?
            else {
                return Ok(false);
            };
            let mut am = row.into_active_model();
            am.cancel_requested = Set(true);
            am.status = Set("cancelling".to_string());
            am.update(&self.db).await?;
            Ok(true)
        }

        async fn finish(&self, run_id: &str) -> Result<()> {
            self.seqs.lock().expect("seqs mutex").remove(run_id);
            if let Some(row) = agent_runs::Entity::find()
                .filter(agent_runs::Column::RunId.eq(run_id))
                .one(&self.db)
                .await?
            {
                // Don't overwrite a cancelling/terminal status with "complete".
                if !matches!(row.status.as_str(), "cancelling" | "cancelled" | "errored") {
                    let mut am = row.into_active_model();
                    am.status = Set("complete".to_string());
                    am.update(&self.db).await?;
                }
            }
            Ok(())
        }
    }
}

#[cfg(feature = "with-db")]
pub use db::{run_hub, DbRunHub};

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::protocol::{AguiEvent, RunOutcome};

    fn text(delta: &str) -> AguiEvent {
        AguiEvent::TextMessageContent {
            message_id: "m1".into(),
            delta: delta.into(),
        }
    }

    #[tokio::test]
    async fn replays_buffered_then_tails_live() {
        let hub = InMemoryRunHub::default();
        hub.start("r1").await.unwrap();
        hub.publish("r1", &text("a")).await.unwrap();
        hub.publish("r1", &text("b")).await.unwrap();

        // Resume from seq 1 -> should see seq 2 (b) on replay, then live seq 3.
        let mut stream = hub.subscribe("r1", 1).await.unwrap();
        hub.publish("r1", &text("c")).await.unwrap();

        let first = stream.next().await.unwrap();
        assert_eq!(first.seq, 2);
        assert_eq!(first.data["delta"], "b");
        let second = stream.next().await.unwrap();
        assert_eq!(second.seq, 3);
        assert_eq!(second.data["delta"], "c");
    }

    #[tokio::test]
    async fn subscribe_from_zero_replays_all() {
        let hub = InMemoryRunHub::default();
        hub.start("r1").await.unwrap();
        hub.publish("r1", &text("a")).await.unwrap();
        hub.publish("r1", &text("b")).await.unwrap();
        let mut stream = hub.subscribe("r1", 0).await.unwrap();
        assert_eq!(stream.next().await.unwrap().seq, 1);
        assert_eq!(stream.next().await.unwrap().seq, 2);
    }

    #[tokio::test]
    async fn cancel_flips_token() {
        let hub = InMemoryRunHub::default();
        let handle = hub.start("r1").await.unwrap();
        assert!(!handle.cancel.is_cancelled());
        assert!(hub.cancel("r1").await.unwrap());
        assert!(handle.cancel.is_cancelled());
        // Unknown run.
        assert!(!hub.cancel("nope").await.unwrap());
    }

    #[tokio::test]
    async fn hub_sink_publishes() {
        let hub: Arc<dyn RunHub> = Arc::new(InMemoryRunHub::default());
        hub.start("r1").await.unwrap();
        let sink = HubSink::new(hub.clone(), "r1");
        sink.emit(AguiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r1".into(),
            outcome: RunOutcome::Success,
            interrupt: None,
        })
        .await
        .unwrap();
        let mut stream = hub.subscribe("r1", 0).await.unwrap();
        assert_eq!(stream.next().await.unwrap().name, "RUN_FINISHED");
    }
}
