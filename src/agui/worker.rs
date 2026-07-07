//! # Durable, worker-driven runs
//!
//! By default a run is driven on a `tokio::spawn` task inside the web process
//! (see [`spawn_inline`]). That survives a dropped client connection — the run
//! publishes to the [run hub](super::hub), and the client resumes via
//! `GET .../stream` — but it does **not** survive a restart of that process.
//!
//! When `agui.execution` is `worker`, the controller instead enqueues a
//! [`RunAgentJob`] on the background-worker queue ([`crate::bgworker`]). A worker
//! picks it up and drives the run through [`execute`], publishing to the
//! (DB-backed) hub. Because the hub streams through the `agent_events` table,
//! the web node that took the request keeps streaming to the client while a
//! different node produces. The run is now durable: it survives a web-process
//! restart and can be retried by the queue.
//!
//! ## Wiring (generated into an app)
//!
//! The controller enqueues via [`RunAgentJob::perform_later`]; the app registers
//! the worker in `connect_workers`, handing it the agent registry:
//!
//! ```rust,ignore
//! async fn connect_workers(ctx: &AppContext, queue: &Queue) -> Result<()> {
//!     queue
//!         .register(loco_rs::agui::worker::RunAgentJob::with_registry(
//!             ctx,
//!             std::sync::Arc::new(crate::agents::registry()),
//!         ))
//!         .await?;
//!     Ok(())
//! }
//! ```
//!
//! Worker execution requires `workers.mode: BackgroundQueue`, a configured
//! queue, and a DB-backed `agui.hub` (`redis`/`postgres`) so the producing
//! worker and the streaming web node share events.

use std::sync::Arc;

use async_trait::async_trait;
use sea_orm::{ActiveModelTrait, ColumnTrait, EntityTrait, QueryFilter, Set};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::agent::{AgentCtx, AgentRegistry, Principal};
use super::artifact::builtin_artifact_tools;
use super::context::{ArtifactStore, MemoryStore};
use super::context_tool::builtin_context_tools;
use super::entities::conversations;
use super::hub::{run_hub, HubSink};
use super::interact::builtin_interact_tools;
use super::memory::builtin_memory_tools;
use super::protocol::{AguiEvent, RunAgentInput};
use super::runtime::{resume, run_turn, ConversationStore, RunParams};
use super::service;
use super::state_tool::builtin_state_tools;
use super::store::{DbArtifactStore, DbMemoryStore, DbStore};
use super::subagent::CompositeToolExecutor;
use super::transport::EventSink;
use crate::app::AppContext;
use crate::bgworker::BackgroundWorker;
use crate::config::{ExecutionConfig, WorkerMode};
use crate::{Error, Result};

/// The serializable payload identifying a run — the durable job body, and the
/// argument shared by the inline and worker execution paths so both drive a run
/// identically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunArgs {
    /// Public id (`pid`) of the conversation the run belongs to.
    pub conversation_pid: String,
    /// The run id (the hub key; the SSE stream's identity).
    pub run_id: String,
    /// The AG-UI input (a fresh message and/or resume instructions).
    pub input: RunAgentInput,
    /// The authenticated principal, captured request-side (there is no request
    /// in a worker), so prompt assembly and authorization behave the same.
    pub principal: Principal,
}

/// Persist the triggering user message for a fresh turn. A resume (which carries
/// no new message, only approve/deny answers) seeds nothing. Kept in the shared
/// executor so the HTTP and headless paths seed identically — callers hand the
/// message on [`RunArgs::input`] and never touch the store themselves.
async fn seed_turn(store: &DbStore, input: &RunAgentInput) -> Result<()> {
    if input.resume.is_empty() {
        if let Some(text) = &input.message {
            store.append_user_message(text).await?;
        }
    }
    Ok(())
}

/// Drive a run to completion: resolve the agent from `registry`, seed the
/// triggering message, build its context / prompt / tools / store, then run the
/// loop publishing to the run hub. Finalizes by finishing the hub and clearing
/// the conversation's active run. Shared by [`spawn_inline`] and [`RunAgentJob`].
///
/// The conversation's `active_run_id` is expected to already be set by the
/// caller ([`dispatch_run`]) before dispatch, so a client that reconnects to
/// `GET .../stream` sees the run immediately. The user message, by contrast, is
/// seeded here so it lands whether the run is inline or picked up by a worker.
///
/// # Errors
/// Propagates a failure to resolve the conversation/agent, seed the message, or
/// build the system prompt. A failure inside the run loop is logged and returned
/// after the hub is finished.
pub async fn execute(
    ctx: &AppContext,
    registry: &AgentRegistry,
    args: &RunArgs,
    cancel: CancellationToken,
) -> Result<()> {
    let uuid =
        Uuid::parse_str(&args.conversation_pid).map_err(|e| Error::Message(e.to_string()))?;
    let conversation = conversations::Entity::find()
        .filter(conversations::Column::Pid.eq(uuid))
        .one(&ctx.db)
        .await?
        .ok_or(Error::NotFound)?;
    let agent = registry
        .get(&conversation.agent_id)
        .ok_or(Error::NotFound)?;

    let store = DbStore::new(ctx.db.clone(), conversation.id);
    seed_turn(&store, &args.input).await?;

    // Rebuild all request-scoped dependencies here, on the *executing* node,
    // from the app context + the (serialized) principal + the persisted
    // conversation row. Nothing below crosses a durable job payload, so inline
    // and worker execution behave identically and tokens are always fresh.
    let actx = AgentCtx {
        app: ctx,
        thread_id: conversation.pid.to_string(),
        conversation_id: conversation.id,
        mode: conversation.mode.clone(),
        principal: args.principal.clone(),
        scope: conversation.scope.clone(),
        extensions: agent.extensions(ctx, &args.principal),
    };
    let mut system = agent.system_prompt(&actx).await?;
    if let Some(plan) = agent.planner(&actx) {
        system.push_str("\n\n# Planning\n");
        system.push_str(&plan);
    }
    let authz = agent.authorizer(&actx);
    let tokens = agent.token_resolver(&actx);
    let embedder = agent.embedder(&actx);
    let guardrail = agent.guardrail(&actx);
    let budget = agent.budget(&actx);
    let hooks = agent.hooks();
    let mut provider = service::provider(ctx, &agent.model());
    // Structured output: constrain the answer to the agent's response schema.
    // A no-op on providers without structured output (e.g. the stub).
    if let Some(schema) = agent.response_schema(&actx) {
        provider.set_response_format(schema);
    }

    let hub = run_hub(ctx);
    let sink: Arc<HubSink> = Arc::new(HubSink::new(hub.clone(), args.run_id.clone()));

    // The run's tool context: app deps, principal, scope, token resolver, the
    // (hub) event sink so tools can emit `CUSTOM` events, and the conversation's
    // artifact store.
    let artifacts: Arc<dyn ArtifactStore> =
        Arc::new(DbArtifactStore::new(ctx.db.clone(), conversation.id));
    let memory: Arc<dyn MemoryStore> = Arc::new(DbMemoryStore::new(
        ctx.db.clone(),
        conversation.scope.clone(),
        Some(conversation.id),
        embedder,
    ));
    let tool_ctx = actx
        .tool_context(args.run_id.clone())
        .with_tokens(tokens)
        .with_sink(sink.clone())
        .with_artifacts(artifacts)
        .with_memory(memory);

    // The agent's own tools plus the framework's built-in artifact / context /
    // memory tools. App tools win on any name collision (first-registered wins).
    let tools = Arc::new(
        CompositeToolExecutor::default()
            .with(agent.tools())
            .with(builtin_artifact_tools())
            .with(builtin_context_tools())
            .with(builtin_memory_tools())
            .with(builtin_state_tools())
            .with(builtin_interact_tools()),
    );

    // Auto-title a fresh conversation from its first user message (cheap
    // heuristic — no extra model call).
    if conversation.title.is_none() {
        if let Some(msg) = &args.input.message {
            let title: String = msg.trim().chars().take(60).collect();
            if !title.is_empty() {
                let _ = service::set_title(&ctx.db, conversation.id, &title).await;
            }
        }
    }

    // Stream the current shared state so a connecting client can render it
    // before the run produces any deltas.
    if let Some(state) = conversation.state.clone() {
        let _ = sink
            .emit(AguiEvent::StateSnapshot { snapshot: state })
            .await;
    }

    let params = RunParams {
        system,
        run_id: args.run_id.clone(),
        thread_id: conversation.pid.to_string(),
        agent: agent.name().to_string(),
        auto_approve: false,
        max_tool_turns: 8,
        tool_timeout: None,
        hooks,
        tool_ctx,
        guardrail,
        budget,
        cancel,
    };

    let result = if let Some(item) = args.input.resume.first().cloned() {
        resume(
            &store,
            tools,
            &provider,
            sink.as_ref(),
            &params,
            &authz,
            &item,
        )
        .await
    } else {
        run_turn(&store, tools, &provider, sink.as_ref(), &params, &authz).await
    };
    if let Err(err) = &result {
        tracing::error!(target: "loco_rs::agui", error = %err, run_id = %args.run_id, "agent run failed");
    }
    let _ = hub.finish(&args.run_id).await;
    let _ = service::clear_active_run(&ctx.db, conversation.id).await;
    result
}

/// Drive a run on an in-process background task (the inline execution path).
/// `cancel` is the token from the caller's `hub.start(run_id)` handle.
pub fn spawn_inline(
    ctx: AppContext,
    registry: Arc<AgentRegistry>,
    args: RunArgs,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let _ = execute(&ctx, &registry, &args, cancel).await;
    });
}

/// Start a run on an existing conversation: register it in the hub, mark it as
/// the conversation's active run, then drive it — durably onto the background
/// queue when `agui.execution=worker` (and a `BackgroundQueue` is configured),
/// otherwise inline. Shared by the HTTP controller and [`start_run`].
///
/// The triggering message is not persisted here — [`execute`] seeds it — so the
/// same call works whether the run is picked up now (inline) or later (worker).
///
/// # Errors
/// Propagates hub / DB errors, or an enqueue failure in worker mode.
pub async fn dispatch_run(
    ctx: &AppContext,
    registry: &Arc<AgentRegistry>,
    conversation_id: i32,
    args: RunArgs,
) -> Result<()> {
    let hub = run_hub(ctx);
    let handle = hub.start(&args.run_id).await?;
    service::set_active_run(&ctx.db, conversation_id, Some(&args.run_id)).await?;

    let execution = ctx
        .config
        .agui
        .as_ref()
        .map(|a| a.execution.clone())
        .unwrap_or_default();
    match execution {
        ExecutionConfig::Inline => {
            spawn_inline(ctx.clone(), registry.clone(), args, handle.cancel);
        }
        // Durable: enqueue so the run outlives this process. Worker execution
        // only makes sense with a real queue — if the app is not in
        // `BackgroundQueue` mode, `perform_later` would run the job with an
        // empty registry, so fall back to inline instead of failing the run.
        ExecutionConfig::Worker if ctx.config.workers.mode == WorkerMode::BackgroundQueue => {
            RunAgentJob::perform_later(ctx, args).await?;
        }
        ExecutionConfig::Worker => {
            tracing::warn!(
                target: "loco_rs::agui",
                "agui.execution=worker requires workers.mode=BackgroundQueue; running inline"
            );
            spawn_inline(ctx.clone(), registry.clone(), args, handle.cancel);
        }
    }
    Ok(())
}

/// A headless run that was started: the new conversation's public id and the run
/// id, so a caller can later read the result from `messages` or attach to the
/// stream (`GET .../stream`).
#[derive(Debug, Clone)]
pub struct StartedRun {
    /// Public id (`pid`) of the conversation opened for the run.
    pub conversation_pid: String,
    /// The run id.
    pub run_id: String,
    /// Numeric id of the opened conversation.
    pub conversation_id: i32,
}

/// Start an agent run with **no HTTP request and no client attached** — from a
/// task, the scheduler, or another worker (generate a report, run a nightly
/// digest, ...).
///
/// Opens a fresh conversation for `agent_id`, hands it `message`, and dispatches
/// the run: durably onto the background-worker queue when `agui.execution=worker`
/// (see [`dispatch_run`]), otherwise inline. Returns the ids so the caller can
/// read the assistant's reply from `messages` afterward — or, more usefully for
/// a report, have the agent's tool write the artifact and treat that as the
/// deliverable.
///
/// ```rust,ignore
/// let run = loco_rs::agui::start_run(
///     &ctx,
///     &std::sync::Arc::new(crate::agents::registry()),
///     "report_writer",
///     "Generate the Q3 sales report",
///     Principal::default(),
///     None, // or Some(json!({ "organization_id": 42 })) to tenant the run
/// ).await?;
/// // later: read `messages` for conversation `run.conversation_pid`
/// ```
///
/// `scope` is the tenancy value stamped on the opened conversation (the same
/// value a request-driven [`ScopeResolver`](super::scope::ScopeResolver) would
/// produce), so headless/offline runs are tenanted identically to HTTP ones and
/// the executing node reads it back off the row.
///
/// # Errors
/// Returns [`Error::NotFound`] if `agent_id` is not in the registry; otherwise
/// propagates DB / hub / enqueue errors.
pub async fn start_run(
    ctx: &AppContext,
    registry: &Arc<AgentRegistry>,
    agent_id: &str,
    message: impl Into<String>,
    principal: Principal,
    scope: Option<serde_json::Value>,
) -> Result<StartedRun> {
    if registry.get(agent_id).is_none() {
        return Err(Error::NotFound);
    }
    let pid = Uuid::new_v4();
    let conversation = conversations::ActiveModel {
        pid: Set(pid),
        agent_id: Set(agent_id.to_string()),
        status: Set(Some("idle".to_string())),
        scope: Set(scope),
        ..Default::default()
    }
    .insert(&ctx.db)
    .await?;

    let run_id = Uuid::new_v4().to_string();
    let args = RunArgs {
        conversation_pid: pid.to_string(),
        run_id: run_id.clone(),
        input: RunAgentInput {
            run_id: None,
            message: Some(message.into()),
            resume: Vec::new(),
        },
        principal,
    };
    dispatch_run(ctx, registry, conversation.id, args).await?;
    Ok(StartedRun {
        conversation_pid: pid.to_string(),
        run_id,
        conversation_id: conversation.id,
    })
}

/// The durable run job. Register one instance per process via
/// [`with_registry`](RunAgentJob::with_registry); the controller enqueues jobs
/// against it with [`perform_later`](BackgroundWorker::perform_later).
#[derive(Clone)]
pub struct RunAgentJob {
    ctx: AppContext,
    registry: Arc<AgentRegistry>,
}

impl RunAgentJob {
    /// Build the worker with the app's agent registry. Use this in
    /// `connect_workers` — the registry cannot be reconstructed from job args,
    /// so the running instance carries it.
    #[must_use]
    pub fn with_registry(ctx: &AppContext, registry: Arc<AgentRegistry>) -> Self {
        Self {
            ctx: ctx.clone(),
            registry,
        }
    }
}

#[async_trait]
impl BackgroundWorker<RunArgs> for RunAgentJob {
    /// Constructs the worker with an **empty** registry. Only the queue's
    /// `perform_later` (which enqueues by class name and never calls `build`)
    /// and the [`with_registry`](RunAgentJob::with_registry)-registered instance
    /// are on the durable path; `build` exists solely to satisfy the trait for
    /// the non-queue `perform_later` modes, which the worker execution path does
    /// not use.
    fn build(ctx: &AppContext) -> Self {
        Self {
            ctx: ctx.clone(),
            registry: Arc::new(AgentRegistry::new()),
        }
    }

    async fn perform(&self, args: RunArgs) -> Result<()> {
        // Take ownership of a poll-backed cancellation token on this node (the
        // row already exists — the web node created it before enqueuing).
        let handle = run_hub(&self.ctx).start(&args.run_id).await?;
        execute(&self.ctx, &self.registry, &args, handle.cancel).await
    }
}

#[cfg(all(test, feature = "agui", feature = "with-db"))]
mod tests {
    use sea_orm::{EntityTrait, PaginatorTrait};
    use sea_orm_migration::SchemaManager;

    use super::{seed_turn, DbStore, RunAgentInput};
    use crate::agui::entities::messages;
    use crate::agui::protocol::{ResumeItem, ResumePayload};
    use crate::schema::{create_table, ColType};

    // Create just the `messages` table (SQLite does not enforce the FK, so the
    // conversation row is unnecessary) so we can exercise `seed_turn` in
    // isolation, with no provider / network.
    async fn store_with_messages_table() -> DbStore {
        let ctx = crate::tests_cfg::app::get_app_context().await;
        let m = SchemaManager::new(&ctx.db);
        create_table(
            &m,
            "messages",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("conversation_id", ColType::IntegerNull),
                ("role", ColType::String),
                ("content", ColType::TextNull),
                ("parts", ColType::JsonBinaryNull),
                ("provider", ColType::StringNull),
                ("model", ColType::StringNull),
                ("usage", ColType::JsonBinaryNull),
                ("status", ColType::StringNull),
            ],
            &[],
        )
        .await
        .expect("create messages table");
        DbStore::new(ctx.db, 1)
    }

    fn fresh(message: &str) -> RunAgentInput {
        RunAgentInput {
            run_id: None,
            message: Some(message.to_string()),
            resume: Vec::new(),
        }
    }

    #[tokio::test]
    async fn seeds_user_message_on_fresh_turn() {
        let store = store_with_messages_table().await;
        seed_turn(&store, &fresh("Generate the Q3 report"))
            .await
            .unwrap();

        let rows = messages::Entity::find().all(&store.db).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].role, "user");
        assert_eq!(rows[0].content.as_deref(), Some("Generate the Q3 report"));
    }

    #[tokio::test]
    async fn does_not_seed_on_resume() {
        let store = store_with_messages_table().await;
        let resume = RunAgentInput {
            run_id: None,
            message: None,
            resume: vec![ResumeItem {
                interrupt_id: "i1".to_string(),
                payload: ResumePayload {
                    approved: true,
                    input: None,
                },
            }],
        };
        seed_turn(&store, &resume).await.unwrap();
        assert_eq!(messages::Entity::find().count(&store.db).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn does_not_seed_empty_message() {
        let store = store_with_messages_table().await;
        let empty = RunAgentInput {
            run_id: None,
            message: None,
            resume: Vec::new(),
        };
        seed_turn(&store, &empty).await.unwrap();
        assert_eq!(messages::Entity::find().count(&store.db).await.unwrap(), 0);
    }
}
