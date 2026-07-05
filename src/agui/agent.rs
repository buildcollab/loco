//! # Agents, a registry, and lifecycle hooks
//!
//! Higher-level, app-facing scaffolding on top of the generic run-loop. An
//! [`Agent`] bundles everything the controller needs to drive one assistant:
//! its identity (the declared **name is its id**), model, system-prompt
//! assembly, typed [`Tools`], lifecycle [`AgentHooks`], and per-call
//! authorization. Register agents in an [`AgentRegistry`] and resolve them by
//! id.
//!
//! Hooks mirror the OpenAI Agents SDK `RunHooks`/`AgentHooks`: observation and
//! side-effect insertion points around the run, each LLM turn, and each tool
//! call. They are **not** the security seam — [`ToolAuthorizer`] still decides
//! whether a call may run (and runs first); [`AgentHooks::before_tool`] fires
//! only after a call is authorized.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::agui::context::{Embedder, NoEmbedder, NoTokens, TokenResolver};
use crate::agui::guardrail::{BudgetLimiter, Guardrail, NoGuardrail, Unlimited};
use crate::agui::provider::{ToolCallReq, TurnOutcome};
use crate::agui::runtime::{AllowAll, ToolAuthorizer};
use crate::agui::tool::Tools;
use crate::app::AppContext;
use crate::{Error, Result};

/// The authenticated principal driving a run, captured request-side and handed
/// to prompt assembly / authorization. Kept minimal and app-agnostic.
///
/// Serializable so it can be carried on a durable
/// [worker job](crate::agui::worker) payload when a run is enqueued.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Principal {
    /// Scopes/permissions the caller holds (e.g. from a JWT `scopes` claim).
    pub scopes: Vec<String>,
    /// Raw claims / app-defined context for richer policies.
    pub claims: Value,
}

impl Principal {
    /// Whether the principal holds `scope`.
    #[must_use]
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

/// Request-scoped context handed to an [`Agent`] for prompt assembly and
/// authorization. Carries the app context (for DB access), the conversation
/// thread id, its mode, and the principal.
pub struct AgentCtx<'a> {
    /// The application context (DB, config, cable, ...).
    pub app: &'a AppContext,
    /// Public id of the conversation (AG-UI `thread_id`).
    pub thread_id: String,
    /// Numeric id of the conversation (for store/artifact scoping).
    pub conversation_id: i32,
    /// Selected conversation mode, if any.
    pub mode: Option<String>,
    /// The authenticated caller.
    pub principal: Principal,
    /// The persisted tenancy value (org/project/...), read from the conversation
    /// row. `None` when the conversation is unscoped. Threaded into the
    /// [`ToolContext`](crate::agui::context::ToolContext) for scoping/billing.
    pub scope: Option<serde_json::Value>,
    /// App-defined custom dependencies, built by [`Agent::extensions`] on the
    /// executing node and forwarded onto the
    /// [`ToolContext`](crate::agui::context::ToolContext). Downcast with
    /// [`ToolContext::ext`](crate::agui::context::ToolContext::ext).
    pub extensions: Arc<dyn std::any::Any + Send + Sync>,
}

/// Lightweight, app-agnostic context passed to [`AgentHooks`] from inside the
/// run-loop. Hooks that need heavier dependencies (DB, user) capture them at
/// construction — the same injection pattern as [`ToolAuthorizer`] and
/// [`ConversationStore`](crate::agui::runtime::ConversationStore).
#[derive(Debug, Clone)]
pub struct RunCtx {
    /// The run id (AG-UI `run_id`).
    pub run_id: String,
    /// The conversation id (AG-UI `thread_id`).
    pub thread_id: String,
    /// The agent id (== [`Agent::name`]).
    pub agent: String,
    /// The model driving this run.
    pub model: String,
}

/// Lifecycle callbacks around a run. Every method defaults to a no-op, so
/// implementors override only what they need.
///
/// Firing order within a run:
/// `on_run_start` → (`before_message` → `after_message`, with
/// `before_tool`/`after_tool` around each executed tool call in between)* →
/// `on_run_end`. `on_error` fires if the run terminates with an error.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Before the run's first LLM turn.
    async fn on_run_start(&self, _ctx: &RunCtx) -> Result<()> {
        Ok(())
    }
    /// After the run reaches a terminal (non-interrupt) state.
    async fn on_run_end(&self, _ctx: &RunCtx) -> Result<()> {
        Ok(())
    }
    /// Before each LLM turn (provider streaming round).
    async fn before_message(&self, _ctx: &RunCtx) -> Result<()> {
        Ok(())
    }
    /// After each LLM turn, with the assembled outcome.
    async fn after_message(&self, _ctx: &RunCtx, _outcome: &TurnOutcome) -> Result<()> {
        Ok(())
    }
    /// Immediately before an (authorized) tool call executes.
    async fn before_tool(&self, _ctx: &RunCtx, _call: &ToolCallReq) -> Result<()> {
        Ok(())
    }
    /// Immediately after a tool call returns its result.
    async fn after_tool(&self, _ctx: &RunCtx, _call: &ToolCallReq, _result: &Value) -> Result<()> {
        Ok(())
    }
    /// The run failed. Advisory only — the loop has already surfaced `RUN_ERROR`.
    async fn on_error(&self, _ctx: &RunCtx, _err: &Error) {}
}

/// A no-op [`AgentHooks`] — the default when an agent declares none.
pub struct NoopHooks;

#[async_trait]
impl AgentHooks for NoopHooks {}

/// A code-declared agent. The declared [`name`](Agent::name) is the agent's id
/// — it is stored on the conversation and used to resolve the agent from the
/// [`AgentRegistry`].
#[async_trait]
pub trait Agent: Send + Sync {
    /// Stable id / registry key for this agent.
    fn name(&self) -> &str;

    /// Human-readable description (surfaced when listing agents).
    fn description(&self) -> &str {
        ""
    }

    /// Default model id for this agent (config may override).
    fn model(&self) -> String;

    /// Assemble the system prompt for a run.
    ///
    /// # Errors
    /// Propagates any error while gathering prompt material (e.g. DB reads).
    async fn system_prompt(&self, ctx: &AgentCtx<'_>) -> Result<String>;

    /// The typed tools this agent exposes.
    fn tools(&self) -> Tools;

    /// Lifecycle hooks for this agent (defaults to no-op).
    fn hooks(&self) -> Arc<dyn AgentHooks> {
        Arc::new(NoopHooks)
    }

    /// Per-call authorization policy (defaults to allow-all; the built-in
    /// write/approval gate still applies).
    fn authorizer(&self, _ctx: &AgentCtx<'_>) -> Arc<dyn ToolAuthorizer> {
        Arc::new(AllowAll)
    }

    /// The token resolver this agent's tools use to obtain access tokens for
    /// external services (defaults to [`NoTokens`]). Built on the executing node
    /// from `ctx` so long-running / worker runs mint fresh tokens rather than
    /// replaying a captured (expired) one.
    fn token_resolver(&self, _ctx: &AgentCtx<'_>) -> Arc<dyn TokenResolver> {
        Arc::new(NoTokens)
    }

    /// The embedder backing this agent's long-term memory search (defaults to
    /// [`NoEmbedder`], i.e. lexical ranking). Return an embedder that calls your
    /// embedding model to enable semantic (cosine) retrieval.
    fn embedder(&self, _ctx: &AgentCtx<'_>) -> Arc<dyn Embedder> {
        Arc::new(NoEmbedder)
    }

    /// The guardrail applied around each turn — inspect/rewrite the model input
    /// and output, or block the run (defaults to a no-op [`NoGuardrail`]).
    fn guardrail(&self, _ctx: &AgentCtx<'_>) -> Arc<dyn Guardrail> {
        Arc::new(NoGuardrail)
    }

    /// The per-turn budget limiter (defaults to [`Unlimited`]). Cap spend per
    /// tenant/run using the run's scope + accumulated usage.
    fn budget(&self, _ctx: &AgentCtx<'_>) -> Arc<dyn BudgetLimiter> {
        Arc::new(Unlimited)
    }

    /// App-defined custom dependencies to place on the run's
    /// [`ToolContext`](crate::agui::context::ToolContext). Return your own deps
    /// struct as `Arc<dyn Any + Send + Sync>`; tools recover it with
    /// [`ToolContext::ext`](crate::agui::context::ToolContext::ext). Defaults to
    /// an empty unit. Built on the executing node, so it is multi-node safe (no
    /// value crosses a serialized job payload).
    fn extensions(
        &self,
        _app: &AppContext,
        _principal: &Principal,
    ) -> Arc<dyn std::any::Any + Send + Sync> {
        Arc::new(())
    }
}

/// A registry mapping agent id → [`Agent`]. Built once at startup and shared.
#[derive(Default, Clone)]
pub struct AgentRegistry {
    agents: HashMap<String, Arc<dyn Agent>>,
}

impl AgentRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent under its [`name`](Agent::name).
    pub fn register<A: Agent + 'static>(&mut self, agent: A) -> &mut Self {
        self.agents
            .insert(agent.name().to_string(), Arc::new(agent));
        self
    }

    /// Resolve an agent by id.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Agent>> {
        self.agents.get(name).cloned()
    }

    /// All registered agent ids.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// All registered agents.
    #[must_use]
    pub fn all(&self) -> Vec<Arc<dyn Agent>> {
        self.agents.values().cloned().collect()
    }
}
