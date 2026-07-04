//! # `ConversationStore` over the agent tables
//!
//! [`DbStore`] maps the framework-owned agent entities
//! ([`entities`](super::entities)) onto the run-loop's
//! [`ConversationStore`](super::runtime::ConversationStore) contract. It is
//! **library** code shared by every agent: the generator no longer emits it —
//! an app only needs the tables (via the generated migration) and this store is
//! constructed for it by the framework controller.

use async_trait::async_trait;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel, QueryFilter,
    Set,
};
use serde_json::{json, Value};
use uuid::Uuid;

use super::entities::{conversations, messages, tool_calls};
use super::provider::{history_from_parts, ChatMessage, ToolCallReq, Usage};
use super::runtime::{ConversationStore, MessageRef, PendingToolCall, ToolRef};
use crate::{Error, Result};

/// Persistence for a single conversation, keyed by its numeric id.
pub struct DbStore {
    /// The database connection (from `AppContext::db`).
    pub db: DatabaseConnection,
    /// The conversation this store reads and writes.
    pub conversation_id: i32,
}

impl DbStore {
    /// Build a store bound to `conversation_id` on `db`.
    #[must_use]
    pub fn new(db: DatabaseConnection, conversation_id: i32) -> Self {
        Self {
            db,
            conversation_id,
        }
    }

    async fn find_message(&self, pid: &str) -> Result<messages::Model> {
        let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
        messages::Entity::find()
            .filter(messages::Column::Pid.eq(uuid))
            .one(&self.db)
            .await?
            .ok_or(Error::NotFound)
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
        // Lossless: rebuild tool_use / tool_result context from the persisted
        // `parts`, falling back to plain `content` for rows without parts.
        Ok(history_from_parts(
            rows.into_iter().map(|m| (m.role, m.parts, m.content)),
        ))
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
        Ok(MessageRef {
            id: pid.to_string(),
        })
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
        Ok(MessageRef {
            id: pid.to_string(),
        })
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
        Ok(ToolRef {
            id: call.id.clone(),
        })
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
            .ok_or(Error::NotFound)?;
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
            .ok_or(Error::NotFound)?;
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
            .ok_or(Error::NotFound)?;
        let mut item = row.into_active_model();
        item.status = Set(Some(status.to_string()));
        item.update(&self.db).await?;
        Ok(())
    }
}
