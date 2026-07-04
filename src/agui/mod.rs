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
//! `cargo loco generate agent <name>` scaffolds all of this into `src/agents/`
//! plus a thin controller. The run handler, in outline:
//!
//! ```rust,ignore
//! let agent = agents::registry().get(&conversation.agent_id)?;      // resolve by id
//! let hub = runtime::run_hub(&ctx);                                 // in-mem or DB-backed
//! let handle = hub.start(&run_id).await?;                           // buffer + cancel token
//! let params = RunParams {
//!     system, run_id, thread_id, agent: agent.name().into(),
//!     hooks: agent.hooks(), cancel: handle.cancel.clone(), ..Default::default()
//! };
//! tokio::spawn(async move {                                         // decoupled from the connection
//!     let sink = HubSink::new(hub.clone(), run_id.clone());         // publishes to the hub
//!     let _ = run_turn(&store, Arc::new(agent.tools()), &provider, &sink, &params, &authz).await;
//!     hub.finish(&run_id).await.ok();
//! });
//! Ok(hub_sse_response(hub.subscribe(&run_id, 0).await?).into_response())  // tail (or resume with ?since=N)
//! ```

pub mod agent;
pub mod hub;
pub mod protocol;
pub mod provider;
pub mod runtime;
pub mod subagent;
pub mod tool;
pub mod transport;

// Flat re-exports for ergonomic `use loco_rs::agui::{...}`.
pub use agent::{
    Agent, AgentCtx, AgentHooks, AgentRegistry, NoopHooks, Principal, RunCtx,
};
pub use hub::{
    channel_stream, in_memory, HubEvent, HubEventStream, HubSink, InMemoryRunHub, RunHandle,
    RunHub, DEFAULT_BUFFER_CAP,
};
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
