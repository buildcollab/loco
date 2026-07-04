//! # `loco_rs::agui` — streaming AI agents over the AG-UI protocol
//!
//! Framework plumbing for exposing LLM agents to a frontend using the
//! [AG-UI](https://docs.ag-ui.com) protocol over Server-Sent Events. This
//! module is **generic infrastructure with zero business logic** — no app
//! concepts (agents lists, modes, DB tables, personas) live here. Everything
//! app-specific arrives through the [`runtime::ConversationStore`],
//! [`runtime::ToolExecutor`], [`runtime::ToolAuthorizer`], and
//! [`provider::Provider`] traits.
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
//!   run-loop, plus the [`runtime::ToolAuthorizer`] per-call authorization seam
//!   ([`runtime::AllowAll`] opts out).
//!
//! ## Higher-level scaffolding
//!
//! [`agent`] adds an app-facing layer on top of the run-loop: declare an
//! [`Agent`](agent::Agent) (its name is its id) with typed [`Tools`](tool::Tools)
//! and lifecycle [`AgentHooks`](agent::AgentHooks); register agents in an
//! [`AgentRegistry`](agent::AgentRegistry). [`hub`] provides the [`RunHub`] that
//! decouples a run from its HTTP connection so streams are **resumable** (a
//! reconnecting client replays from a sequence number) and **cancellable** (an
//! explicit stop flips the run's [`CancellationToken`]).
//!
//! ## DB-backed subsystem (behind `with-db`)
//!
//! The persistence and HTTP wiring that used to be *generated into every app*
//! now lives here as library code, so a project only writes agent-specific
//! declarations:
//!
//! - [`entities`] — the framework-owned SeaORM entities for the agent tables.
//! - [`store::DbStore`] — the [`ConversationStore`](runtime::ConversationStore)
//!   over those tables.
//! - [`hub::DbRunHub`] + [`run_hub`](hub::run_hub) — the multi-node run hub and
//!   its config-driven selection.
//! - [`service`] — the config-driven [`provider`](service::provider) +
//!   [`assemble_system`](service::assemble_system) factories.
//! - [`controller::routes`] — the reusable HTTP router (list / open / run /
//!   stream / cancel).
//! - [`worker`] — durable, background-worker-driven runs (`agui.execution`).
//!
//! `cargo loco generate agent <name>` now scaffolds only the migration and the
//! per-agent modules (prompt / tools / hooks) plus a one-line controller that
//! mounts [`controller::routes`] and a one-line worker registration. The run
//! handler in outline (see [`controller`]):
//!
//! ```rust,ignore
//! let agent = registry.get(&conversation.agent_id)?;               // resolve by id
//! let hub = run_hub(&ctx);                                         // in-mem or DB-backed
//! let handle = hub.start(&run_id).await?;                          // buffer + cancel token
//! service::set_active_run(&ctx.db, conv.id, Some(&run_id)).await?; // resumable/cancellable
//! match execution {                                               // agui.execution
//!     Inline => worker::spawn_inline(ctx.clone(), registry, args, handle.cancel),
//!     Worker => worker::RunAgentJob::perform_later(&ctx, args).await?, // durable
//! }
//! Ok(hub_sse_response(hub.subscribe(&run_id, 0).await?).into_response())  // tail (or ?since=N)
//! ```

pub mod agent;
pub mod hub;
pub mod protocol;
pub mod provider;
pub mod runtime;
pub mod subagent;
pub mod tool;
pub mod transport;

// DB-backed pieces (previously generated into every app): the framework-owned
// entities, the `ConversationStore`, config-driven factories, the reusable HTTP
// router, and the durable worker. Enabled together with the `with-db` feature.
#[cfg(feature = "with-db")]
pub mod controller;
#[cfg(feature = "with-db")]
pub mod entities;
#[cfg(feature = "with-db")]
pub mod service;
#[cfg(feature = "with-db")]
pub mod store;
#[cfg(feature = "with-db")]
pub mod worker;

// Flat re-exports for ergonomic `use loco_rs::agui::{...}`.
pub use agent::{
    Agent, AgentCtx, AgentHooks, AgentRegistry, NoopHooks, Principal, RunCtx,
};
pub use hub::{
    channel_stream, in_memory, HubEvent, HubEventStream, HubSink, InMemoryRunHub, RunHandle,
    RunHub, DEFAULT_BUFFER_CAP,
};
#[cfg(feature = "with-db")]
pub use hub::{run_hub, DbRunHub};
#[cfg(feature = "with-db")]
pub use service::{assemble_system, clear_active_run, provider as config_provider, set_active_run};
#[cfg(feature = "with-db")]
pub use store::DbStore;
#[cfg(feature = "with-db")]
pub use worker::{execute, spawn_inline, RunAgentJob, RunArgs};
// Re-exported so generated app code can build cancellation tokens / run hubs
// without adding `tokio-util` as a direct dependency.
pub use tokio_util::sync::CancellationToken;
pub use protocol::{
    part_text, part_tool_result, part_tool_use, AguiEvent, Interrupt, ResumeItem, ResumePayload,
    RunAgentInput, RunOutcome,
};
pub use provider::{
    history_from_parts, AgentDelta, ChatMessage, Provider, RigConfig, RigProvider, StreamAssembler,
    StubProvider, ToolCallReq, ToolKind, ToolSpec, TurnOutcome, Usage, OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
};
pub use tool::{NoArgs, Tool, Tools};
pub use runtime::{
    resume, resume_with_subagents, run_turn, run_turn_with_subagents, AllowAll, ConversationStore,
    MessageRef, PendingToolCall, RunParams, ToolAuthorizer, ToolDecision, ToolExecutor, ToolRef,
};
pub use subagent::{
    default_task_schema, CompositeToolExecutor, InMemoryStore, LocalSubagent, Subagent,
    SubagentCtx, SubagentExecutor, SubagentOutput, SubagentRegistry, SubagentStep,
    DEFAULT_MAX_SUBAGENT_DEPTH,
};
pub use transport::{
    event_to_sse, hub_event_to_sse, hub_sse_response, sse_response, spawn_and_stream, EventSink,
    MpscSink, NullSink,
};
