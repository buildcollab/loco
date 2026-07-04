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
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::agent::{AgentCtx, AgentRegistry, Principal};
use super::entities::conversations;
use super::hub::{run_hub, HubSink};
use super::runtime::{resume, run_turn, RunParams};
use super::service;
use super::store::DbStore;
use crate::app::AppContext;
use crate::bgworker::BackgroundWorker;
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
    pub input: super::protocol::RunAgentInput,
    /// The authenticated principal, captured request-side (there is no request
    /// in a worker), so prompt assembly and authorization behave the same.
    pub principal: Principal,
}

/// Drive a run to completion: resolve the agent from `registry`, build its
/// context / prompt / tools / store, then run the loop publishing to the run
/// hub. Finalizes by finishing the hub and clearing the conversation's active
/// run. Shared by [`spawn_inline`] and [`RunAgentJob`].
///
/// The user message (for a fresh turn) and the conversation's `active_run_id`
/// are expected to already be persisted by the caller before dispatch, so a
/// client that reconnects to `GET .../stream` sees the run immediately.
///
/// # Errors
/// Propagates a failure to resolve the conversation/agent or to build the
/// system prompt. A failure inside the run loop is logged and returned after
/// the hub is finished.
pub async fn execute(
    ctx: &AppContext,
    registry: &AgentRegistry,
    args: &RunArgs,
    cancel: CancellationToken,
) -> Result<()> {
    let uuid = Uuid::parse_str(&args.conversation_pid).map_err(|e| Error::Message(e.to_string()))?;
    let conversation = conversations::Entity::find()
        .filter(conversations::Column::Pid.eq(uuid))
        .one(&ctx.db)
        .await?
        .ok_or(Error::NotFound)?;
    let agent = registry.get(&conversation.agent_id).ok_or(Error::NotFound)?;

    let actx = AgentCtx {
        app: ctx,
        thread_id: conversation.pid.to_string(),
        mode: conversation.mode.clone(),
        principal: args.principal.clone(),
    };
    let system = agent.system_prompt(&actx).await?;
    let authz = agent.authorizer(&actx);
    let hooks = agent.hooks();
    let tools = Arc::new(agent.tools());
    let store = DbStore::new(ctx.db.clone(), conversation.id);
    let provider = service::provider(ctx, &agent.model());

    let params = RunParams {
        system,
        run_id: args.run_id.clone(),
        thread_id: conversation.pid.to_string(),
        agent: agent.name().to_string(),
        auto_approve: false,
        max_tool_turns: 8,
        tool_timeout: None,
        hooks,
        cancel,
    };

    let hub = run_hub(ctx);
    let sink = HubSink::new(hub.clone(), args.run_id.clone());
    let result = if let Some(item) = args.input.resume.first().cloned() {
        resume(&store, tools, &provider, &sink, &params, &authz, &item).await
    } else {
        run_turn(&store, tools, &provider, &sink, &params, &authz).await
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
