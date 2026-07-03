to: src/controllers/agents.rs
skip_exists: true
message: |
  Agent controller `agents` was added.

  Next steps:
    1. Enable the `agui` feature on loco-rs in Cargo.toml:
         loco-rs = { version = "*", features = ["agui"] }
    2. Run the migration + entity sync:
         $ cargo loco db migrate && cargo loco db entities
    3. Set your OpenRouter key in the environment:
         export OPENROUTER_API_KEY=sk-or-...
    4. Seed an `agents` row (name, provider, model, system_prompt) and POST to
         /api/conversations/{conversation_pid}/run
       with an AG-UI `RunAgentInput` body to stream a response. The `run`
       endpoint requires a JWT; write tools additionally require a `memo:write`
       entry in the token's `scopes` claim (see `AppAuthorizer`).

  This file is a starting point — customize the tools in `AgentTools`, the
  authorization policy in `AppAuthorizer`, the system-prompt assembly, and the
  message-history mapping to fit your app.
injections:
- into: src/controllers/mod.rs
  append: true
  content: "pub mod agents;"
- into: src/app.rs
  after: "AppRoutes::"
  content: "            .add_route(controllers::agents::routes())"
---
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unused_async)]
//! AG-UI agent chat controller.
//!
//! Exposes endpoints to list agents, open conversations, post context, and
//! stream agent responses over the AG-UI protocol (SSE). The heavy lifting
//! lives in `loco_rs::agui`; this file wires your database tables to its
//! `ConversationStore` + `ToolExecutor` traits and its `ToolAuthorizer`
//! per-tool-call authorization seam.

use async_trait::async_trait;
use loco_rs::agui::{
    run_turn, resume, spawn_and_stream, ChatMessage, ConversationStore, MessageRef,
    PendingToolCall, RigProvider, RunAgentInput, RunParams, ToolAuthorizer, ToolCallReq,
    ToolDecision, ToolExecutor, ToolKind, ToolRef, ToolSpec, Usage,
};
use loco_rs::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::models::_entities::{
    agent_modes, agents, context_items, conversations, messages, tool_calls,
};

// ---------------------------------------------------------------------------
// Tool executor — example tools. Replace with your own.
// ---------------------------------------------------------------------------

struct AgentTools;

#[async_trait]
impl ToolExecutor for AgentTools {
    fn specs(&self) -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "get_time".to_string(),
                description: "Return the current server time (example read tool).".to_string(),
                parameters: json!({ "type": "object", "properties": {} }),
                kind: ToolKind::Read,
            },
            ToolSpec {
                name: "save_memo".to_string(),
                description: "Persist a short memo (example write tool, gated by human approval)."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
                kind: ToolKind::Write,
            },
        ]
    }

    async fn execute(&self, name: &str, args: Value) -> Result<Value> {
        match name {
            "get_time" => Ok(json!({ "time": chrono::Utc::now().to_rfc3339() })),
            "save_memo" => Ok(json!({
                "saved": true,
                "text": args.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
            other => Err(Error::Message(format!("unknown tool: {other}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool authorization — checked for every tool call before it runs. Replace the
// example policy with your own.
// ---------------------------------------------------------------------------

/// Scope a caller must hold to invoke a write tool in this example policy.
const WRITE_SCOPE: &str = "memo:write";

/// Per-call authorization for tool calls. Carries the authenticated principal's
/// scopes (populated from `auth::JWT` claims in the `run` handler below); tune
/// the policy in [`AppAuthorizer::authorize`] to fit your app.
struct AppAuthorizer {
    /// Scopes the caller holds, read from the JWT `scopes` claim.
    scopes: Vec<String>,
}

impl AppAuthorizer {
    fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

#[async_trait]
impl ToolAuthorizer for AppAuthorizer {
    async fn authorize(&self, call: &ToolCallReq, kind: ToolKind) -> Result<ToolDecision> {
        // Read tools are unrestricted in this example.
        if kind == ToolKind::Read {
            return Ok(ToolDecision::Allow);
        }

        // Write tools require an explicit scope. Callers without it are refused
        // outright — the model sees the denial as the tool result and can
        // respond. Callers with it fall through to the built-in human-approval
        // gate (or run immediately when `RunParams::auto_approve` is set).
        //
        // The three decisions available here are `ToolDecision::Deny`,
        // `ToolDecision::RequireApproval` (force approval even for a read tool),
        // and `ToolDecision::Allow`.
        if self.has_scope(WRITE_SCOPE) {
            Ok(ToolDecision::Allow)
        } else {
            Ok(ToolDecision::Deny {
                reason: format!("missing scope '{WRITE_SCOPE}' for tool '{}'", call.name),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// OPTIONAL: subagents with local tool calls.
// ---------------------------------------------------------------------------
//
// A subagent is a child agent the model can delegate a task to. It runs its own
// turn-loop with its own *local* (in-process) `ToolExecutor`, streams its events
// to its own sink (e.g. persisted to a table for review), and returns its final
// text to the parent as a tool result. Expose subagents to the parent by
// composing a `SubagentExecutor` with `AgentTools` via `CompositeToolExecutor`.
//
// This is commented out so the generated controller compiles as-is. To enable,
// add the imports:
//     use loco_rs::agui::{
//         CompositeToolExecutor, LocalSubagent, RigProvider, SubagentExecutor,
//         SubagentRegistry, AllowAll, EventSink, AguiEvent,
//     };
//     use std::sync::Arc;
//
// A DB-logging sink for subagent runs (persist each event for debugging/review):
//
//     struct DbLoggingSink { db: DatabaseConnection, conversation_id: i32 }
//     #[async_trait]
//     impl EventSink for DbLoggingSink {
//         async fn emit(&self, ev: AguiEvent) -> Result<()> {
//             // INSERT a row: (conversation_id, event_name, serde_json::to_value(&ev))
//             let _ = (&self.db, self.conversation_id, ev);
//             Ok(())
//         }
//     }
//
// Build the composite executor inside the `run` handler and pass it to
// `run_turn`/`resume` instead of `AgentTools`:
//
//     let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
//     let mut registry = SubagentRegistry::default();
//     registry.register(LocalSubagent {
//         name: "summarizer".to_string(),
//         description: "Summarize the provided text into a few sentences.".to_string(),
//         system: "You are a concise summarizer.".to_string(),
//         provider: RigProvider::new(api_key.clone(), None, agent.model.clone()),
//         exec: AgentTools,      // the subagent's own local tools
//         authz: AllowAll,       // the subagent's own authorization policy
//         max_tool_turns: 4,
//     });
//     let sink = Arc::new(DbLoggingSink { db: ctx.db.clone(), conversation_id: conversation.id });
//     let exec = CompositeToolExecutor::default()
//         .with(AgentTools)
//         .with(SubagentExecutor::new(Arc::new(registry), sink));
//     // ... then `run_turn(&store, &exec, &provider, &sink, &params, &authz)`.
//
// Note: subagents backed by the ephemeral in-memory store auto-approve their
// own write tools. A subagent whose write tools should require *human* approval
// that bubbles up to the parent needs a persistent child-conversation store so
// its state survives the interrupt→resume round-trip.

// ---------------------------------------------------------------------------
// Conversation store — maps the agent tables to the agui run-loop.
// ---------------------------------------------------------------------------

struct DbStore {
    db: DatabaseConnection,
    conversation_id: i32,
}

impl DbStore {
    async fn find_message(&self, pid: &str) -> Result<messages::Model> {
        let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
        messages::Entity::find()
            .filter(messages::Column::Pid.eq(uuid))
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)
    }
}

#[async_trait]
impl ConversationStore for DbStore {
    async fn load_history(&self) -> Result<Vec<ChatMessage>> {
        let mut rows = messages::Entity::find()
            .filter(messages::Column::ConversationId.eq(self.conversation_id))
            .all(&self.db)
            .await?;
        rows.sort_by_key(|m| m.id);
        // NOTE: this maps role + text only. If you use multi-turn tool calls,
        // reconstruct assistant `tool_calls` and `tool` results here from the
        // persisted `parts` / `tool_calls` rows so the model sees full context.
        Ok(rows
            .into_iter()
            .map(|m| ChatMessage::text(&m.role, &m.content.unwrap_or_default()))
            .collect())
    }

    async fn append_user_message(&self, text: &str) -> Result<MessageRef> {
        let pid = Uuid::new_v4();
        let item = messages::ActiveModel {
            pid: Set(pid),
            conversation_id: Set(self.conversation_id),
            role: Set("user".to_string()),
            content: Set(Some(text.to_string())),
            status: Set(Some("complete".to_string())),
            ..Default::default()
        };
        item.insert(&self.db).await?;
        Ok(MessageRef { id: pid.to_string() })
    }

    async fn begin_assistant_message(&self, provider: &str, model: &str) -> Result<MessageRef> {
        let pid = Uuid::new_v4();
        let item = messages::ActiveModel {
            pid: Set(pid),
            conversation_id: Set(self.conversation_id),
            role: Set("assistant".to_string()),
            provider: Set(Some(provider.to_string())),
            model: Set(Some(model.to_string())),
            status: Set(Some("streaming".to_string())),
            ..Default::default()
        };
        item.insert(&self.db).await?;
        Ok(MessageRef { id: pid.to_string() })
    }

    async fn record_tool_call(
        &self,
        msg: &MessageRef,
        call: &ToolCallReq,
        status: &str,
    ) -> Result<ToolRef> {
        let message = self.find_message(&msg.id).await?;
        let item = tool_calls::ActiveModel {
            pid: Set(Uuid::new_v4()),
            message_id: Set(message.id),
            tool_call_id: Set(call.id.clone()),
            name: Set(call.name.clone()),
            arguments: Set(Some(call.arguments.clone())),
            status: Set(status.to_string()),
            ..Default::default()
        };
        item.insert(&self.db).await?;
        Ok(ToolRef { id: call.id.clone() })
    }

    async fn complete_tool_call(
        &self,
        tool: &ToolRef,
        status: &str,
        result: &Value,
        duration_ms: i64,
    ) -> Result<()> {
        let row = tool_calls::Entity::find()
            .filter(tool_calls::Column::ToolCallId.eq(&tool.id))
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;
        let mut item = row.into_active_model();
        item.status = Set(status.to_string());
        item.result = Set(Some(result.clone()));
        item.duration_ms = Set(Some(duration_ms));
        item.update(&self.db).await?;
        Ok(())
    }

    async fn finalize_assistant_message(
        &self,
        msg: &MessageRef,
        parts: Value,
        usage: &Usage,
        status: &str,
    ) -> Result<()> {
        let usage_json = json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
        });
        let mut item = self.find_message(&msg.id).await?.into_active_model();
        item.parts = Set(Some(parts));
        item.usage = Set(Some(usage_json));
        item.status = Set(Some(status.to_string()));
        item.update(&self.db).await?;
        Ok(())
    }

    async fn find_pending_tool_call(&self, tool_call_id: &str) -> Result<Option<PendingToolCall>> {
        let row = tool_calls::Entity::find()
            .filter(tool_calls::Column::ToolCallId.eq(tool_call_id))
            .filter(tool_calls::Column::Status.eq("pending"))
            .one(&self.db)
            .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let message = messages::Entity::find_by_id(row.message_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;
        Ok(Some(PendingToolCall {
            tool_call_id: row.tool_call_id,
            name: row.name,
            arguments: row.arguments.unwrap_or_else(|| json!({})),
            message_id: message.pid.to_string(),
        }))
    }

    async fn set_conversation_status(&self, status: &str) -> Result<()> {
        let row = conversations::Entity::find_by_id(self.conversation_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;
        let mut item = row.into_active_model();
        item.status = Set(Some(status.to_string()));
        item.update(&self.db).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn find_agent(ctx: &AppContext, pid: &str) -> Result<agents::Model> {
    let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
    agents::Entity::find()
        .filter(agents::Column::Pid.eq(uuid))
        .one(&ctx.db)
        .await?
        .ok_or_else(|| Error::NotFound)
}

async fn find_conversation(ctx: &AppContext, pid: &str) -> Result<conversations::Model> {
    let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
    conversations::Entity::find()
        .filter(conversations::Column::Pid.eq(uuid))
        .one(&ctx.db)
        .await?
        .ok_or_else(|| Error::NotFound)
}

/// Build the system prompt from the agent, the selected mode, and any context
/// items attached to the conversation.
async fn assemble_system(
    ctx: &AppContext,
    agent: &agents::Model,
    mode: Option<&str>,
    conversation_id: i32,
) -> Result<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(sp) = &agent.system_prompt {
        parts.push(sp.clone());
    }
    if let Some(mode_name) = mode {
        if let Some(m) = agent_modes::Entity::find()
            .filter(agent_modes::Column::AgentId.eq(agent.id))
            .filter(agent_modes::Column::Name.eq(mode_name))
            .one(&ctx.db)
            .await?
        {
            if let Some(sp) = m.system_prompt {
                parts.push(sp);
            }
        }
    }
    let items = context_items::Entity::find()
        .filter(context_items::Column::ConversationId.eq(conversation_id))
        .all(&ctx.db)
        .await?;
    for item in items {
        if let Some(content) = item.content {
            parts.push(format!("# Context: {}\n{content}", item.name));
        }
    }
    Ok(parts.join("\n\n"))
}

// ---------------------------------------------------------------------------
// Request bodies
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

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[debug_handler]
pub async fn list_agents(State(ctx): State<AppContext>) -> Result<Response> {
    format::json(agents::Entity::find().all(&ctx.db).await?)
}

#[debug_handler]
pub async fn get_agent(
    Path(agent_pid): Path<String>,
    State(ctx): State<AppContext>,
) -> Result<Response> {
    format::json(find_agent(&ctx, &agent_pid).await?)
}

#[debug_handler]
pub async fn list_conversations(
    Path(agent_pid): Path<String>,
    State(ctx): State<AppContext>,
) -> Result<Response> {
    let agent = find_agent(&ctx, &agent_pid).await?;
    let rows = conversations::Entity::find()
        .filter(conversations::Column::AgentId.eq(agent.id))
        .all(&ctx.db)
        .await?;
    format::json(rows)
}

#[debug_handler]
pub async fn create_conversation(
    Path(agent_pid): Path<String>,
    State(ctx): State<AppContext>,
    Json(params): Json<CreateConversationParams>,
) -> Result<Response> {
    let agent = find_agent(&ctx, &agent_pid).await?;
    let mode = params.mode.or_else(|| agent.default_mode.clone());
    let item = conversations::ActiveModel {
        pid: Set(Uuid::new_v4()),
        agent_id: Set(agent.id),
        title: Set(params.title),
        mode: Set(mode),
        status: Set(Some("idle".to_string())),
        ..Default::default()
    };
    format::json(item.insert(&ctx.db).await?)
}

#[debug_handler]
pub async fn list_messages(
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

#[debug_handler]
pub async fn add_context(
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

/// Stream an agent turn (or resume an approval) over AG-UI SSE.
#[debug_handler]
pub async fn run(
    Path(conversation_pid): Path<String>,
    State(ctx): State<AppContext>,
    auth: auth::JWT,
    Json(input): Json<RunAgentInput>,
) -> Result<Response> {
    // Scopes the caller holds, read from the JWT `scopes` claim (a JSON array of
    // strings). These drive `AppAuthorizer`'s per-tool-call decisions.
    let scopes: Vec<String> = auth
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

    let conversation = find_conversation(&ctx, &conversation_pid).await?;
    let agent = agents::Entity::find_by_id(conversation.agent_id)
        .one(&ctx.db)
        .await?
        .ok_or_else(|| Error::NotFound)?;

    let system = assemble_system(&ctx, &agent, conversation.mode.as_deref(), conversation.id).await?;

    let store = DbStore {
        db: ctx.db.clone(),
        conversation_id: conversation.id,
    };

    // Persist the incoming user message for a fresh turn (resumes carry none).
    if input.resume.is_empty() {
        if let Some(text) = &input.message {
            store.append_user_message(text).await?;
        }
    }

    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let provider = RigProvider::new(api_key, None, agent.model.clone());
    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let params = RunParams {
        system,
        run_id,
        thread_id: conversation.pid.to_string(),
        auto_approve: false,
        max_tool_turns: 8,
    };
    let resume_item = input.resume.first().cloned();

    let sse = spawn_and_stream(64, || {}, move |sink| async move {
        let exec = AgentTools;
        // Authorize every tool call against the caller's scopes.
        let authz = AppAuthorizer { scopes };
        let result = if let Some(item) = resume_item {
            resume(&store, &exec, &provider, &sink, &params, &authz, &item).await
        } else {
            run_turn(&store, &exec, &provider, &sink, &params, &authz).await
        };
        if let Err(err) = result {
            tracing::error!(error = %err, "agent run failed");
        }
    });

    Ok(sse.into_response())
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("api/")
        .add("agents", get(list_agents))
        .add("agents/{agent_pid}", get(get_agent))
        .add("agents/{agent_pid}/conversations", get(list_conversations))
        .add("agents/{agent_pid}/conversations", post(create_conversation))
        .add("conversations/{conversation_pid}/messages", get(list_messages))
        .add("conversations/{conversation_pid}/context", post(add_context))
        .add("conversations/{conversation_pid}/run", post(run))
}
