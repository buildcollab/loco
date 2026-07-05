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

use axum::extract::{Extension, Multipart, Path, Query, State};
use sea_orm::{ActiveModelTrait, ColumnTrait, Condition, EntityTrait, QueryFilter, Set};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use super::agent::{AgentRegistry, Principal};
use super::entities::{artifacts, context_items, conversations, messages};
use super::hub::run_hub;
use super::protocol::RunAgentInput;
use super::scope::{NoScope, ScopeResolver};
use super::transport::hub_sse_response;
use super::worker::{dispatch_run, RunArgs};
use crate::app::AppContext;
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

/// Resolve a conversation by `pid`, applying the request's tenancy `filter` (if
/// any) so a caller cannot reach a conversation outside its scope — the single
/// choke point every conversation-resolving handler goes through.
async fn find_conversation(
    ctx: &AppContext,
    pid: &str,
    filter: Option<&Condition>,
) -> Result<conversations::Model> {
    let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
    let mut query =
        conversations::Entity::find().filter(conversations::Column::Pid.eq(uuid));
    if let Some(cond) = filter {
        query = query.filter(cond.clone());
    }
    query.one(&ctx.db).await?.ok_or(Error::NotFound)
}

/// The request's resolved tenancy scope: the JSON `value` to stamp on a new
/// conversation, and the DB `filter` restricting which conversations the request
/// may read. Produced by the mounted [`ScopeResolver`] (default [`NoScope`] →
/// both `None`, i.e. unscoped).
struct ReqScope {
    value: Option<Value>,
    filter: Option<Condition>,
}

impl<S> axum::extract::FromRequestParts<S> for ReqScope
where
    AppContext: axum::extract::FromRef<S>,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> std::result::Result<Self, Self::Rejection> {
        use axum::extract::FromRef;
        let ctx = AppContext::from_ref(state);
        // Principal extraction is infallible (missing/invalid token → anonymous).
        let principal =
            <PrincipalExtract as axum::extract::FromRequestParts<S>>::from_request_parts(
                parts, state,
            )
            .await
            .unwrap_or(PrincipalExtract(Principal::default()))
            .0;
        let resolver = parts
            .extensions
            .get::<Arc<dyn ScopeResolver>>()
            .cloned()
            .unwrap_or_else(|| Arc::new(NoScope) as Arc<dyn ScopeResolver>);
        let value = resolver
            .resolve(&ctx, parts, &principal)
            .await
            .map_err(IntoResponse::into_response)?;
        let filter = value.as_ref().map(|s| resolver.filter(s));
        Ok(Self { value, filter })
    }
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

async fn get_agent(
    Path(agent_id): Path<String>,
    Extension(registry): Registry,
) -> Result<Response> {
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
    scope: ReqScope,
) -> Result<Response> {
    let mut query =
        conversations::Entity::find().filter(conversations::Column::AgentId.eq(&agent_id));
    if let Some(cond) = &scope.filter {
        query = query.filter(cond.clone());
    }
    let rows = query.all(&ctx.db).await?;
    format::json(rows)
}

async fn create_conversation(
    Path(agent_id): Path<String>,
    State(ctx): State<AppContext>,
    Extension(registry): Registry,
    scope: ReqScope,
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
        // Stamp the request's tenancy scope so every later run/read is scoped.
        scope: Set(scope.value),
        ..Default::default()
    };
    format::json(item.insert(&ctx.db).await?)
}

async fn list_messages(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    scope: ReqScope,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
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
    scope: ReqScope,
    Json(params): Json<ContextParams>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
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

/// Upload a file as a conversation context item. The bytes are written to shared
/// [`Storage`](crate::storage) (so any executing node can fetch them) and a
/// `kind="file"` context item is created whose `reference` is the storage key.
async fn upload_context(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    scope: ReqScope,
    mut multipart: Multipart,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
    let mut created: Vec<Value> = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| Error::Message(e.to_string()))?
    {
        let name = field
            .file_name()
            .or_else(|| field.name())
            .unwrap_or("upload")
            .to_string();
        let content_type = field.content_type().map(String::from);
        let bytes = field
            .bytes()
            .await
            .map_err(|e| Error::Message(e.to_string()))?;
        let size = bytes.len();
        // Namespaced, collision-free storage key under the conversation.
        let key = format!(
            "agui/{}/context/{}-{}",
            conversation.pid,
            Uuid::new_v4(),
            name
        );
        ctx.storage
            .upload(std::path::Path::new(&key), &bytes)
            .await?;
        let item = context_items::ActiveModel {
            pid: Set(Uuid::new_v4()),
            conversation_id: Set(conversation.id),
            kind: Set("file".to_string()),
            name: Set(name),
            reference: Set(Some(key)),
            metadata: Set(Some(json!({ "mime": content_type, "size": size }))),
            ..Default::default()
        };
        created.push(serde_json::to_value(item.insert(&ctx.db).await?)?);
    }
    format::json(json!({ "uploaded": created }))
}

/// List a conversation's artifacts (for display). Scoped through the same
/// tenancy filter as every other conversation read.
async fn list_artifacts(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    scope: ReqScope,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
    let mut rows = artifacts::Entity::find()
        .filter(artifacts::Column::ConversationId.eq(conversation.id))
        .all(&ctx.db)
        .await?;
    rows.sort_by_key(|a| a.id);
    format::json(rows)
}

/// Fetch a single artifact of a conversation by its `pid`.
async fn get_artifact(
    Path((conversation_pid, artifact_pid)): Path<(String, String)>,
    State(ctx): State<AppContext>,
    scope: ReqScope,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
    let uuid = Uuid::parse_str(&artifact_pid).map_err(|e| Error::Message(e.to_string()))?;
    let row = artifacts::Entity::find()
        .filter(artifacts::Column::Pid.eq(uuid))
        .filter(artifacts::Column::ConversationId.eq(conversation.id))
        .one(&ctx.db)
        .await?
        .ok_or(Error::NotFound)?;
    format::json(row)
}

/// Start (or resume-approve) a run. The run is decoupled from this connection:
/// it streams into the run hub, so a dropped connection does not stop it — the
/// client resumes via `GET /stream`. This response tails the run from seq 0.
///
/// Execution is inline (`tokio::spawn`) or handed to a durable background worker
/// depending on `agui.execution` — [`dispatch_run`] handles both; this handler
/// only resolves the conversation and then subscribes.
async fn run(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    Extension(registry): Registry,
    principal: PrincipalExtract,
    scope: ReqScope,
    Json(input): Json<RunAgentInput>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
    // Validate the agent id against the registry before starting anything.
    if registry.get(&conversation.agent_id).is_none() {
        return Err(Error::NotFound);
    }

    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let args = RunArgs {
        conversation_pid: conversation.pid.to_string(),
        run_id: run_id.clone(),
        input,
        principal: principal.0,
    };
    dispatch_run(&ctx, &registry, conversation.id, args).await?;

    let stream = run_hub(&ctx).subscribe(&run_id, 0).await?;
    Ok(hub_sse_response(stream).into_response())
}

/// Resume the live stream of the conversation's active run (network reconnect).
/// Returns 204 when no run is active.
async fn stream(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    scope: ReqScope,
    Query(q): Query<StreamQuery>,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
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
    scope: ReqScope,
) -> Result<Response> {
    let conversation = find_conversation(&ctx, &conversation_pid, scope.filter.as_ref()).await?;
    if let Some(run_id) = &conversation.active_run_id {
        run_hub(&ctx).cancel(run_id).await?;
    }
    format::empty()
}

/// Build the agent routes, capturing the app's agent `registry` in an extension
/// layer so the handlers can resolve an agent by id. Conversations are unscoped
/// ([`NoScope`]). Mount with
/// `AppRoutes::add_route(loco_rs::agui::controller::routes(registry))`.
#[must_use]
pub fn routes(registry: Arc<AgentRegistry>) -> Routes {
    routes_with_scope(registry, Arc::new(NoScope))
}

/// Like [`routes`], but tenants conversations with an app-supplied
/// [`ScopeResolver`] (org/project/...): it stamps the scope on create and filters
/// every conversation read, so a request cannot reach a conversation outside its
/// scope.
#[must_use]
pub fn routes_with_scope(
    registry: Arc<AgentRegistry>,
    scope: Arc<dyn ScopeResolver>,
) -> Routes {
    Routes::new()
        .prefix("api/")
        .add("agents", get(list_agents))
        .add("agents/{agent_id}", get(get_agent))
        .add("agents/{agent_id}/conversations", get(list_conversations))
        .add("agents/{agent_id}/conversations", post(create_conversation))
        .add(
            "conversations/{conversation_pid}/messages",
            get(list_messages),
        )
        .add(
            "conversations/{conversation_pid}/context",
            post(add_context),
        )
        .add(
            "conversations/{conversation_pid}/context/upload",
            post(upload_context),
        )
        .add(
            "conversations/{conversation_pid}/artifacts",
            get(list_artifacts),
        )
        .add(
            "conversations/{conversation_pid}/artifacts/{artifact_pid}",
            get(get_artifact),
        )
        .add("conversations/{conversation_pid}/run", post(run))
        .add("conversations/{conversation_pid}/stream", get(stream))
        .add("conversations/{conversation_pid}/cancel", post(cancel))
        .layer(Extension(registry))
        .layer(Extension(scope))
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
