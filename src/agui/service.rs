//! # Runtime wiring factories
//!
//! The config-driven construction that used to be generated into
//! `src/agents/runtime.rs`: building the LLM [`provider`] from `agui.provider`
//! and assembling a system prompt from a conversation's mode and context items.
//! These are **library** helpers so an app only writes agent-specific prompts,
//! not the plumbing that feeds them.

use sea_orm::{
    ActiveModelTrait, ColumnTrait, Condition, DatabaseConnection, EntityTrait, IntoActiveModel,
    QueryFilter, Set,
};
use serde_json::Value;
use uuid::Uuid;

use super::agent::AgentCtx;
use super::entities::{context_items, conversations};
use super::provider::{Provider, RigProvider, StubProvider};
use crate::app::AppContext;
use crate::config::ProviderConfig;
use crate::{Error, Result};

/// Build the LLM provider from `agui.provider` config, defaulting the model to
/// `default_model` (usually the agent's declared model) when config does not
/// override it.
///
/// Returns a boxed [`Provider`] so the concrete backend (real
/// [`RigProvider`] or the network-free [`StubProvider`], selected by
/// `agui.provider.kind`) is a config decision, not a compile-time one — a test
/// or local-dev config can set `kind: stub` to drive the exact production
/// run-loop with no API key or network.
///
/// With no `agui` config, returns an empty-key provider that will fail on first
/// call — configure `agui.provider` in `config/*.yaml`.
#[must_use]
pub fn provider(ctx: &AppContext, default_model: &str) -> Box<dyn Provider> {
    match ctx.config.agui.as_ref() {
        Some(cfg) => {
            if matches!(cfg.provider, ProviderConfig::Stub(_)) {
                return Box::new(StubProvider::new());
            }
            let model = cfg
                .provider
                .settings()
                .default_model
                .clone()
                .unwrap_or_else(|| default_model.to_string());
            Box::new(RigProvider::from_config(&cfg.provider, model))
        }
        None => Box::new(RigProvider::new(
            String::new(),
            None,
            default_model.to_string(),
        )),
    }
}

/// Assemble a system prompt from a base string plus the conversation's mode and
/// any attached context items.
///
/// # Errors
/// Propagates DB errors while loading the conversation and its context items.
pub async fn assemble_system(ctx: &AgentCtx<'_>, base: &str) -> Result<String> {
    let mut parts = vec![base.to_string()];
    if let Ok(uuid) = Uuid::parse_str(&ctx.thread_id) {
        if let Some(conv) = conversations::Entity::find()
            .filter(conversations::Column::Pid.eq(uuid))
            .one(&ctx.app.db)
            .await?
        {
            if let Some(mode) = &ctx.mode {
                parts.push(format!("# Mode: {mode}"));
            }
            let items = context_items::Entity::find()
                .filter(context_items::Column::ConversationId.eq(conv.id))
                .all(&ctx.app.db)
                .await?;
            let mut attachments: Vec<String> = Vec::new();
            for item in items {
                if let Some(content) = item.content {
                    parts.push(format!("# Context: {}\n{content}", item.name));
                } else if item.reference.is_some() {
                    // File/resource with no inline content: tell the model it
                    // exists so it can fetch it via the `read_context` tool.
                    attachments.push(item.name);
                }
            }
            if !attachments.is_empty() {
                parts.push(format!(
                    "# Attachments\nThe following are attached to this conversation; read one with \
                     the `read_context` tool: {}",
                    attachments.join(", ")
                ));
            }
        }
    }
    Ok(parts.join("\n\n"))
}

/// Look up a conversation by its public id, applying an optional tenancy
/// `filter` (from a [`ScopeResolver`](super::scope::ScopeResolver)) so a caller
/// cannot reach a conversation outside its scope. For apps that wire their own
/// controller instead of [`routes`](super::controller::routes).
///
/// # Errors
/// [`Error::NotFound`] if no matching (in-scope) conversation exists.
pub async fn find_conversation(
    db: &DatabaseConnection,
    pid: &str,
    filter: Option<Condition>,
) -> Result<conversations::Model> {
    let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
    let mut query = conversations::Entity::find().filter(conversations::Column::Pid.eq(uuid));
    if let Some(cond) = filter {
        query = query.filter(cond);
    }
    query.one(db).await?.ok_or(Error::NotFound)
}

/// Create a conversation for `agent_id`, stamping the tenancy `scope` the caller
/// resolved from the request (e.g. `{organization_id, project_id}`). For apps
/// that create conversations from their own controller. Returns the new row.
///
/// # Errors
/// Propagates DB errors on insert.
pub async fn create_conversation(
    db: &DatabaseConnection,
    agent_id: &str,
    title: Option<String>,
    mode: Option<String>,
    scope: Option<Value>,
) -> Result<conversations::Model> {
    let item = conversations::ActiveModel {
        pid: Set(Uuid::new_v4()),
        agent_id: Set(agent_id.to_string()),
        title: Set(title),
        mode: Set(mode),
        status: Set(Some("idle".to_string())),
        scope: Set(scope),
        ..Default::default()
    };
    Ok(item.insert(db).await?)
}

/// Point a conversation at its in-flight run so a client can resume or cancel
/// it (`GET .../stream`, `POST .../cancel`). Pass `None` to clear.
///
/// # Errors
/// Propagates DB errors while updating the conversation.
pub async fn set_active_run(
    db: &DatabaseConnection,
    conversation_id: i32,
    run_id: Option<&str>,
) -> Result<()> {
    if let Some(row) = conversations::Entity::find_by_id(conversation_id)
        .one(db)
        .await?
    {
        let mut am = row.into_active_model();
        am.active_run_id = Set(run_id.map(String::from));
        am.update(db).await?;
    }
    Ok(())
}

/// Clear a conversation's active run (run finished / errored / cancelled).
///
/// # Errors
/// Propagates DB errors while updating the conversation.
pub async fn clear_active_run(db: &DatabaseConnection, conversation_id: i32) -> Result<()> {
    set_active_run(db, conversation_id, None).await
}

/// Set a conversation's `title` (used for auto-titling a fresh thread).
///
/// # Errors
/// Propagates DB errors while updating the conversation.
pub async fn set_title(db: &DatabaseConnection, conversation_id: i32, title: &str) -> Result<()> {
    if let Some(row) = conversations::Entity::find_by_id(conversation_id)
        .one(db)
        .await?
    {
        let mut am = row.into_active_model();
        am.title = Set(Some(title.to_string()));
        am.update(db).await?;
    }
    Ok(())
}
