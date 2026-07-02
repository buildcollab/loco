//! # `loco_rs::agui` — streaming AI agents over the AG-UI protocol
//!
//! Framework plumbing for exposing LLM agents to a frontend using the
//! [AG-UI](https://docs.ag-ui.com) protocol over Server-Sent Events. This
//! module is **generic infrastructure with zero business logic** — no app
//! concepts (agents lists, modes, DB tables, personas) live here. Everything
//! app-specific arrives through the [`runtime::ConversationStore`],
//! [`runtime::ToolExecutor`], and [`provider::Provider`] traits.
//!
//! Enable with the `agui` cargo feature.
//!
//! ## Pieces
//!
//! - [`protocol`] — AG-UI wire event types + message-part builders.
//! - [`provider`] — the [`provider::Provider`] LLM abstraction, an
//!   OpenRouter-backed [`provider::RigProvider`], and a network-free
//!   [`provider::StubProvider`].
//! - [`transport`] — an [`transport::EventSink`] plus the SSE response builder
//!   and a `spawn_and_stream` convenience.
//! - [`runtime`] — the reusable [`runtime::run_turn`] / [`runtime::resume`]
//!   run-loop.
//!
//! ## Sketch (an axum handler)
//!
//! ```rust,ignore
//! use loco_rs::agui::{
//!     protocol::RunAgentInput,
//!     provider::RigProvider,
//!     runtime::{run_turn, resume, RunParams},
//!     transport::spawn_and_stream,
//! };
//!
//! async fn run(/* State(ctx), Path(conv), Json(input): RunAgentInput */) -> impl axum::response::IntoResponse {
//!     let provider = RigProvider::new(api_key, None, model);          // OpenRouter
//!     let params = RunParams { system, run_id, thread_id, auto_approve: false, max_tool_turns: 8 };
//!     spawn_and_stream(64, move || { /* clear "responding" status */ }, move |sink| async move {
//!         let store = /* impl ConversationStore for this conversation */;
//!         let exec  = /* impl ToolExecutor */;
//!         let res = if let Some(item) = input.resume.first() {
//!             resume(&store, &exec, &provider, &sink, &params, item).await
//!         } else {
//!             run_turn(&store, &exec, &provider, &sink, &params).await
//!         };
//!         let _ = res; // errors are already surfaced as RUN_ERROR on the sink
//!     })
//! }
//! ```

pub mod protocol;
pub mod provider;
pub mod runtime;
pub mod transport;

// Flat re-exports for ergonomic `use loco_rs::agui::{...}`.
pub use protocol::{
    part_text, part_tool_result, part_tool_use, AguiEvent, Interrupt, ResumeItem, ResumePayload,
    RunAgentInput, RunOutcome,
};
pub use provider::{
    AgentDelta, ChatMessage, Provider, RigProvider, StreamAssembler, StubProvider, ToolCallReq,
    ToolKind, ToolSpec, TurnOutcome, Usage, OPENROUTER_BASE_URL,
};
pub use runtime::{
    resume, run_turn, ConversationStore, MessageRef, PendingToolCall, RunParams, ToolExecutor,
    ToolRef,
};
pub use transport::{event_to_sse, sse_response, spawn_and_stream, EventSink, MpscSink, NullSink};
