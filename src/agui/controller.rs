//! # Reusable agent HTTP router
//!
//! The thin HTTP layer over the agent runtime, moved out of the generator and
//! into the framework. An app mounts it with the registry of its declared
//! agents:
//!
//! ```rust,ignore
//! // src/controllers/agents.rs (generated — the whole file)
//! use loco_rs::prelude::*;
//! pub fn routes() -> Routes {
//!     loco_rs::agui::controller::routes(std::sync::Arc::new(crate::agents::registry()))
//! }
//! ```
//!
//! All agent logic (persistence, provider, run hub, execution) lives in
//! `loco_rs::agui`; this module only wires HTTP → the registry + run hub, and
//! dispatches a run inline or onto the durable worker based on `agui.execution`.
//!
//! Endpoints (under the `api/` prefix):
//!
//! ```text
//! GET  agents                                list declared agents
//! GET  agents/{agent_id}                     one agent
//! GET  agents/{agent_id}/conversations       list conversations
//! POST agents/{agent_id}/conversations       open a conversation
//! GET  conversations/{pid}/messages          message history
//! POST conversations/{pid}/context           attach context
//! POST conversations/{pid}/run               start a run (SSE, resumable)
//! GET  conversations/{pid}/stream?since=N     resume the live stream
//! POST conversations/{pid}/cancel            stop the active run
//! ```

use std::sync::Arc;

use axum::extract::{Extension, Path, Query, State};
use sea_orm::{ActiveModelTrait, ColumnTrait, EntityTrait, QueryFilter, Set};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use super::agent::{AgentRegistry, Principal};
use super::entities::{context_items, conversations, messages};
use super::hub::run_hub;
use super::protocol::RunAgentInput;
use super::runtime::ConversationStore;
use super::store::DbStore;
use super::transport::hub_sse_response;
use super::worker::{spawn_inline, RunAgentJob, RunArgs};
use crate::app::AppContext;
use crate::bgworker::BackgroundWorker;
use crate::config::ExecutionConfig;
use crate::controller::{format, Json, Routes};
use crate::{Error, Result};

use axum::response::IntoResponse;
use axum::routing::{get, post};

type Registry = Extension<Arc<AgentRegistry>>;

// ---------------------------------------------------------------------------
// Request bodies / queries
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateConversationParams {
    pub title: Option<String>,
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContextParams {
    pub kind: String,
    pub name: String,
    pub reference: Option<String>,
    pub content: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct StreamQuery {
    /// Resume from just after this sequence number (the last SSE `id:` seen).
    #[serde(default)]
    pub since: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn find_conversation(ctx: &AppContext, pid: &str) -> Result<conversations::Model> {
    let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
    conversations::Entity::find()
        .filter(conversations::Column::Pid.eq(uuid))
        .one(&ctx.db)
        .await?
        .ok_or(Error::NotFound)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn list_agents(Extension(registry): Registry) -> Result<Response> {
    let agents: Vec<Value> = registry
        .all()
        .into_iter()
        .map(|a| json!({ "name": a.name(), "description": a.description(), "model": a.model() }))
        .collect();
    format::json(agents)
}

async fn get_agent(Path(agent_id): Path<String>, Extension(registry): Registry) -> Result<Response> {
    let agent = registry.get(&agent_id).ok_or(Error::NotFound)?;
    format::json(json!({
        "name": agent.name(),
        "description": agent.description(),
        "model": agent.model(),
    }))
}

async fn list_conversations(
    Path(agent_id): Path<String>,
    State(ctx): State<AppContext>,
) -> Result<Response> {
    let rows = conversations::Entity::find()
        .filter(conversations::Column::AgentId.eq(&agent_id))
        .all(&ctx.db)
        .await?;
    format::json(rows)
}

async fn create_conversation(
    Path(agent_id): Path<String>,
    State(ctx): State<AppContext>,
    Extension(registry): Registry,
    Json(params): Json<CreateConversationParams>,
) -> Result<Response> {
    if registry.get(&agent_id).is_none() {
        return Err(Error::NotFound);
    }
    let item = conversations::ActiveModel {
        pid: Set(Uuid::new_v4()),
        agent_id: Set(agent_id),
        title: Set(params.title),
        mode: Set(params.mode),
        status: Set(Some("idle".to_string())),
        ..Default::default()
    };
    format::json(item.insert(&ctx.db).await?)
}

async fn list_messages(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid).await?;
    let mut rows = messages::Entity::find()
        .filter(messages::Column::ConversationId.eq(conversation.id))
        .all(&ctx.db)
        .await?;
    rows.sort_by_key(|m| m.id);
    format::json(rows)
}

async fn add_context(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    Json(params): Json<ContextParams>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid).await?;
    let item = context_items::ActiveModel {
        pid: Set(Uuid::new_v4()),
        conversation_id: Set(conversation.id),
        kind: Set(params.kind),
        name: Set(params.name),
        reference: Set(params.reference),
        content: Set(params.content),
        metadata: Set(params.metadata),
        ..Default::default()
    };
    format::json(item.insert(&ctx.db).await?)
}

/// Start (or resume-approve) a run. The run is decoupled from this connection:
/// it streams into the run hub, so a dropped connection does not stop it — the
/// client resumes via `GET /stream`. This response tails the run from seq 0.
///
/// Execution is inline (`tokio::spawn`) or handed to a durable background worker
/// depending on `agui.execution`; either way this handler starts the run in the
/// hub, records the active run, persists the user message, then subscribes.
async fn run(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    Extension(registry): Registry,
    principal: PrincipalExtract,
    Json(input): Json<RunAgentInput>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid).await?;
    // Validate the agent id against the registry before starting anything.
    if registry.get(&conversation.agent_id).is_none() {
        return Err(Error::NotFound);
    }

    let store = DbStore::new(ctx.db.clone(), conversation.id);
    // Persist the incoming user message for a fresh turn (resumes carry none).
    if input.resume.is_empty() {
        if let Some(text) = &input.message {
            store.append_user_message(text).await?;
        }
    }

    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let hub = run_hub(&ctx);
    let handle = hub.start(&run_id).await?;
    super::service::set_active_run(&ctx.db, conversation.id, Some(&run_id)).await?;

    let args = RunArgs {
        conversation_pid: conversation.pid.to_string(),
        run_id: run_id.clone(),
        input,
        principal: principal.0,
    };

    let execution = ctx
        .config
        .agui
        .as_ref()
        .map(|a| a.execution.clone())
        .unwrap_or_default();
    match execution {
        ExecutionConfig::Inline => {
            spawn_inline(ctx.clone(), registry, args, handle.cancel);
        }
        // Durable: enqueue so the run outlives this process; a worker calls
        // `hub.start` again (idempotent) for its own cancel token. Worker
        // execution only makes sense with a real queue — if the app is not in
        // `BackgroundQueue` mode, `perform_later` would run the job with an
        // empty registry, so fall back to inline instead of failing the run.
        ExecutionConfig::Worker
            if ctx.config.workers.mode == crate::config::WorkerMode::BackgroundQueue =>
        {
            RunAgentJob::perform_later(&ctx, args).await?;
        }
        ExecutionConfig::Worker => {
            tracing::warn!(
                target: "loco_rs::agui",
                "agui.execution=worker requires workers.mode=BackgroundQueue; running inline"
            );
            spawn_inline(ctx.clone(), registry, args, handle.cancel);
        }
    }

    let stream = hub.subscribe(&run_id, 0).await?;
    Ok(hub_sse_response(stream).into_response())
}

/// Resume the live stream of the conversation's active run (network reconnect).
/// Returns 204 when no run is active.
async fn stream(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    Query(q): Query<StreamQuery>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid).await?;
    let Some(run_id) = conversation.active_run_id.clone() else {
        return Ok((axum::http::StatusCode::NO_CONTENT, ()).into_response());
    };
    let hub = run_hub(&ctx);
    let stream = hub.subscribe(&run_id, q.since.unwrap_or(0)).await?;
    Ok(hub_sse_response(stream).into_response())
}

/// Cancel ("stop") the conversation's active run.
async fn cancel(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid).await?;
    if let Some(run_id) = &conversation.active_run_id {
        run_hub(&ctx).cancel(run_id).await?;
    }
    format::empty()
}

/// Build the agent routes, capturing the app's agent `registry` in an extension
/// layer so the handlers can resolve an agent by id. Mount with
/// `AppRoutes::add_route(loco_rs::agui::controller::routes(registry))`.
#[must_use]
pub fn routes(registry: Arc<AgentRegistry>) -> Routes {
    Routes::new()
        .prefix("api/")
        .add("agents", get(list_agents))
        .add("agents/{agent_id}", get(get_agent))
        .add("agents/{agent_id}/conversations", get(list_conversations))
        .add("agents/{agent_id}/conversations", post(create_conversation))
        .add("conversations/{conversation_pid}/messages", get(list_messages))
        .add("conversations/{conversation_pid}/context", post(add_context))
        .add("conversations/{conversation_pid}/run", post(run))
        .add("conversations/{conversation_pid}/stream", get(stream))
        .add("conversations/{conversation_pid}/cancel", post(cancel))
        .layer(Extension(registry))
}

// ---------------------------------------------------------------------------
// Principal extraction
// ---------------------------------------------------------------------------

use axum::response::Response;

/// Wraps the [`Principal`] driving a run, extracted from a bearer JWT when the
/// `auth_jwt` feature is on (missing/invalid token → an anonymous principal, so
/// the routes are not force-authenticated). Apps can layer their own auth
/// middleware over [`routes`] for stricter policies.
struct PrincipalExtract(Principal);

#[cfg(feature = "auth_jwt")]
impl<S> axum::extract::FromRequestParts<S> for PrincipalExtract
where
    AppContext: axum::extract::FromRef<S>,
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> std::result::Result<Self, Self::Rejection> {
        use crate::controller::extractor::auth;
        let principal = match auth::extract_jwt_from_request_parts(parts, state) {
            Ok(jwt) => {
                let scopes = jwt
                    .claims
                    .claims
                    .get("scopes")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                Principal {
                    scopes,
                    claims: serde_json::to_value(&jwt.claims.claims).unwrap_or(Value::Null),
                }
            }
            Err(_) => Principal::default(),
        };
        Ok(Self(principal))
    }
}

#[cfg(not(feature = "auth_jwt"))]
impl<S> axum::extract::FromRequestParts<S> for PrincipalExtract
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        _parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> std::result::Result<Self, Self::Rejection> {
        Ok(Self(Principal::default()))
    }
}
