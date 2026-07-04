//! # Runtime wiring factories
//!
//! The config-driven construction that used to be generated into
//! `src/agents/runtime.rs`: building the LLM [`provider`] from `agui.provider`
//! and assembling a system prompt from a conversation's mode and context items.
//! These are **library** helpers so an app only writes agent-specific prompts,
//! not the plumbing that feeds them.

use sea_orm::{ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel, QueryFilter, Set};
use uuid::Uuid;

use super::agent::AgentCtx;
use super::entities::{context_items, conversations};
use super::provider::RigProvider;
use crate::app::AppContext;
use crate::Result;

/// Build the LLM provider from `agui.provider` config, defaulting the model to
/// `default_model` (usually the agent's declared model) when config does not
/// override it.
///
/// With no `agui` config, returns an empty-key provider that will fail on first
/// call — configure `agui.provider` in `config/*.yaml`.
#[must_use]
pub fn provider(ctx: &AppContext, default_model: &str) -> RigProvider {
    match ctx.config.agui.as_ref() {
        Some(cfg) => {
            let model = cfg
                .provider
                .settings()
                .default_model
                .clone()
                .unwrap_or_else(|| default_model.to_string());
            RigProvider::from_config(&cfg.provider, model)
        }
        None => RigProvider::new(String::new(), None, default_model.to_string()),
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
            for item in items {
                if let Some(content) = item.content {
                    parts.push(format!("# Context: {}\n{content}", item.name));
                }
            }
        }
    }
    Ok(parts.join("\n\n"))
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
