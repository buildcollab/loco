//! # Built-in context tools
//!
//! Tools that let an agent discover and read the conversation's *context items*
//! — text notes, references to system resources, and uploaded files (see the
//! `context` / `context/upload` controller endpoints). Text is returned inline;
//! a file's bytes are fetched from shared [`Storage`](crate::storage) by its
//! stored `reference` key, so this works on any executing node.
//!
//! Composed into a run by the framework in
//! [`worker::execute`](crate::agui::worker::execute) via
//! [`builtin_context_tools`].

use async_trait::async_trait;
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
use serde::Deserialize;
use serde_json::{json, Value};

use super::context::ToolContext;
use super::entities::context_items;
use super::provider::{ToolKind, ToolSpec};
use super::tool::{NoArgs, Tool, Tools};
use crate::{Error, Result};

/// The framework's built-in context tools: `list_context`, `read_context`.
#[must_use]
pub fn builtin_context_tools() -> Tools {
    Tools::new().with(ListContext).with(ReadContext)
}

fn require_ctx(ctx: &ToolContext) -> Result<(&sea_orm::DatabaseConnection, i32)> {
    let db = ctx
        .app()
        .map(|a| &a.db)
        .ok_or_else(|| Error::string("context tools require app context"))?;
    Ok((db, ctx.conversation_id))
}

/// List the conversation's context items (name, kind, whether inline or a file).
struct ListContext;

#[async_trait]
impl Tool for ListContext {
    type Args = NoArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_context".to_string(),
            description: "List the context items attached to this conversation (text notes, \
                          system-resource references, and uploaded files) so you can read one \
                          with read_context."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, _args: NoArgs) -> Result<Value> {
        let (db, conversation_id) = require_ctx(ctx)?;
        let rows = context_items::Entity::find()
            .filter(context_items::Column::ConversationId.eq(conversation_id))
            .all(db)
            .await?;
        let items: Vec<Value> = rows
            .into_iter()
            .map(|c| {
                json!({
                    "pid": c.pid.to_string(),
                    "name": c.name,
                    "kind": c.kind,
                    "hasContent": c.content.is_some(),
                    "isFile": c.reference.is_some(),
                    "metadata": c.metadata,
                })
            })
            .collect();
        Ok(json!({ "context": items }))
    }
}

#[derive(Deserialize)]
struct ReadArgs {
    /// The context item's name (or its `pid`).
    name: String,
}

/// Read a context item's content by name (or pid). Inline text is returned
/// directly; a file is fetched from storage and returned as UTF-8 text.
struct ReadContext;

#[async_trait]
impl Tool for ReadContext {
    type Args = ReadArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_context".to_string(),
            description: "Read a conversation context item by its name (or id): returns inline \
                          text directly, or fetches an uploaded file's contents from storage."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "name": { "type": "string", "description": "The context item's name or id." } },
                "required": ["name"]
            }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: ReadArgs) -> Result<Value> {
        let (db, conversation_id) = require_ctx(ctx)?;
        let rows = context_items::Entity::find()
            .filter(context_items::Column::ConversationId.eq(conversation_id))
            .all(db)
            .await?;
        let item = rows
            .into_iter()
            .find(|c| c.name == args.name || c.pid.to_string() == args.name)
            .ok_or_else(|| Error::Message(format!("no context item named '{}'", args.name)))?;

        if let Some(content) = item.content {
            return Ok(json!({ "name": item.name, "content": content }));
        }
        if let Some(reference) = item.reference {
            let app = ctx
                .app()
                .ok_or_else(|| Error::string("reading a file requires app context"))?;
            let bytes: Vec<u8> = app
                .storage
                .download(std::path::Path::new(&reference))
                .await?;
            return match String::from_utf8(bytes) {
                Ok(text) => Ok(json!({ "name": item.name, "content": text })),
                Err(e) => Ok(json!({
                    "name": item.name,
                    "reference": reference,
                    "error": "file is not UTF-8 text",
                    "bytes": e.into_bytes().len(),
                })),
            };
        }
        Ok(json!({ "name": item.name, "content": Value::Null }))
    }
}
