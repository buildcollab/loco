//! # `ConversationStore` over the agent tables
//!
//! [`DbStore`] maps the framework-owned agent entities
//! ([`entities`](super::entities)) onto the run-loop's
//! [`ConversationStore`](super::runtime::ConversationStore) contract. It is
//! **library** code shared by every agent: the generator no longer emits it —
//! an app only needs the tables (via the generated migration) and this store is
//! constructed for it by the framework controller.

use std::sync::Arc;

use async_trait::async_trait;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, Condition, DatabaseConnection, EntityTrait, IntoActiveModel,
    QueryFilter, Set,
};
use serde_json::{json, Value};
use uuid::Uuid;

use super::context::{
    Artifact, ArtifactStore, Embedder, MemoryHit, MemoryStore, NewArtifact, NewMemory,
};
use super::entities::{artifacts, conversations, memories, messages, tool_calls};
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

/// [`ArtifactStore`](super::context::ArtifactStore) over the `artifacts` table,
/// scoped to a single conversation. Built by the framework in
/// [`worker::execute`](super::worker::execute) and placed on the run's
/// [`ToolContext`](super::context::ToolContext).
pub struct DbArtifactStore {
    /// The database connection (from `AppContext::db`).
    pub db: DatabaseConnection,
    /// The conversation these artifacts belong to.
    pub conversation_id: i32,
}

impl DbArtifactStore {
    /// Build a store bound to `conversation_id` on `db`.
    #[must_use]
    pub fn new(db: DatabaseConnection, conversation_id: i32) -> Self {
        Self {
            db,
            conversation_id,
        }
    }

    async fn find(&self, pid: &str) -> Result<Option<artifacts::Model>> {
        let uuid = Uuid::parse_str(pid).map_err(|e| Error::Message(e.to_string()))?;
        Ok(artifacts::Entity::find()
            .filter(artifacts::Column::Pid.eq(uuid))
            .filter(artifacts::Column::ConversationId.eq(self.conversation_id))
            .one(&self.db)
            .await?)
    }
}

/// Read a JSON array-of-strings column into a `Vec<String>`.
fn tags_from_json(v: &Option<Value>) -> Vec<String> {
    v.as_ref()
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn to_artifact(m: artifacts::Model) -> Artifact {
    Artifact {
        pid: m.pid.to_string(),
        name: m.name,
        kind: m.kind,
        content: m.content,
        reference: m.reference,
        tags: tags_from_json(&m.tags),
        metadata: m.metadata,
        version: m.version,
    }
}

#[async_trait]
impl ArtifactStore for DbArtifactStore {
    async fn create(&self, new: NewArtifact) -> Result<Artifact> {
        let pid = Uuid::new_v4();
        let item = artifacts::ActiveModel {
            pid: Set(pid),
            conversation_id: Set(self.conversation_id),
            name: Set(new.name),
            kind: Set(new.kind),
            content: Set(new.content),
            reference: Set(new.reference),
            tags: Set(Some(json!(new.tags))),
            metadata: Set(new.metadata),
            version: Set(1),
            ..Default::default()
        };
        Ok(to_artifact(item.insert(&self.db).await?))
    }

    async fn update(
        &self,
        pid: &str,
        content: Option<String>,
        tags: Option<Vec<String>>,
        metadata: Option<Value>,
    ) -> Result<Artifact> {
        let row = self.find(pid).await?.ok_or(Error::NotFound)?;
        let version = row.version + 1;
        let mut item = row.into_active_model();
        if let Some(content) = content {
            item.content = Set(Some(content));
        }
        if let Some(tags) = tags {
            item.tags = Set(Some(json!(tags)));
        }
        if let Some(metadata) = metadata {
            item.metadata = Set(Some(metadata));
        }
        item.version = Set(version);
        Ok(to_artifact(item.update(&self.db).await?))
    }

    async fn get(&self, pid: &str) -> Result<Option<Artifact>> {
        Ok(self.find(pid).await?.map(to_artifact))
    }

    async fn list(&self, tag: Option<&str>) -> Result<Vec<Artifact>> {
        let mut rows = artifacts::Entity::find()
            .filter(artifacts::Column::ConversationId.eq(self.conversation_id))
            .all(&self.db)
            .await?;
        rows.sort_by_key(|a| a.id);
        Ok(rows
            .into_iter()
            .filter(|a| match tag {
                Some(t) => tags_from_json(&a.tags).iter().any(|x| x == t),
                None => true,
            })
            .map(to_artifact)
            .collect())
    }
}

/// [`MemoryStore`](super::context::MemoryStore) over the `memories` table,
/// scoped to a tenant (and optionally a conversation). Embeds content on write
/// when an [`Embedder`] is configured; ranks search by cosine similarity, or by
/// lexical token overlap when no embeddings exist. Candidate rows are ranked
/// in-process (portable across databases) — swap for a pgvector query when scale
/// demands it.
pub struct DbMemoryStore {
    db: DatabaseConnection,
    scope: Option<Value>,
    conversation_id: Option<i32>,
    embedder: Arc<dyn Embedder>,
    /// Safety cap on candidate rows loaded for in-process ranking.
    candidate_cap: u64,
}

impl DbMemoryStore {
    /// Build a memory store for `scope` (tenant) and optional `conversation_id`.
    #[must_use]
    pub fn new(
        db: DatabaseConnection,
        scope: Option<Value>,
        conversation_id: Option<i32>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            db,
            scope,
            conversation_id,
            embedder,
            candidate_cap: 1000,
        }
    }

    /// Rows visible to this store: tenant-scoped (or global `NULL` scope) memory,
    /// plus this conversation's own memory.
    fn visibility(&self) -> Condition {
        let scope_cond = match &self.scope {
            Some(s) => Condition::any()
                .add(memories::Column::Scope.eq(s.clone()))
                .add(memories::Column::Scope.is_null()),
            None => Condition::all().add(memories::Column::Scope.is_null()),
        };
        match self.conversation_id {
            Some(cid) => Condition::any()
                .add(scope_cond)
                .add(memories::Column::ConversationId.eq(cid)),
            None => scope_cond,
        }
    }
}

/// Cosine similarity of two equal-length vectors (0 if degenerate).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Lexical fallback: fraction of the query's distinct lowercased tokens that
/// appear in `text` (0..=1).
fn lexical_score(query: &str, text: &str) -> f32 {
    let qtokens: std::collections::BTreeSet<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(str::to_lowercase)
        .collect();
    if qtokens.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let hits = qtokens.iter().filter(|t| lower.contains(*t)).count();
    hits as f32 / qtokens.len() as f32
}

fn embedding_from_json(v: &Option<Value>) -> Option<Vec<f32>> {
    v.as_ref().and_then(Value::as_array).map(|a| {
        a.iter()
            .filter_map(|x| x.as_f64().map(|f| f as f32))
            .collect()
    })
}

#[async_trait]
impl MemoryStore for DbMemoryStore {
    async fn add(&self, items: Vec<NewMemory>) -> Result<usize> {
        if items.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = items.iter().map(|i| i.content.clone()).collect();
        // Best-effort embedding: if it fails or is empty, store without vectors.
        let embeds = self.embedder.embed(&texts).await.unwrap_or_default();
        let mut count = 0;
        for (i, item) in items.into_iter().enumerate() {
            let embedding = embeds.get(i).map(|v| json!(v));
            let row = memories::ActiveModel {
                pid: Set(Uuid::new_v4()),
                scope: Set(self.scope.clone()),
                conversation_id: Set(self.conversation_id),
                kind: Set(item.kind),
                content: Set(item.content),
                embedding: Set(embedding),
                metadata: Set(item.metadata),
                ..Default::default()
            };
            row.insert(&self.db).await?;
            count += 1;
        }
        Ok(count)
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<MemoryHit>> {
        use sea_orm::QuerySelect;
        let rows = memories::Entity::find()
            .filter(self.visibility())
            .limit(self.candidate_cap)
            .all(&self.db)
            .await?;

        // Embed the query once (empty vec → lexical fallback).
        let qemb = self
            .embedder
            .embed(&[query.to_string()])
            .await
            .ok()
            .and_then(|mut v| v.pop())
            .filter(|v| !v.is_empty());

        let mut scored: Vec<MemoryHit> = rows
            .into_iter()
            .map(|r| {
                let score = match (&qemb, embedding_from_json(&r.embedding)) {
                    (Some(q), Some(e)) if !e.is_empty() => cosine(q, &e),
                    _ => lexical_score(query, &r.content),
                };
                MemoryHit {
                    id: r.pid.to_string(),
                    content: r.content,
                    score,
                    kind: r.kind,
                    metadata: r.metadata,
                }
            })
            .filter(|h| h.score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k.max(1));
        Ok(scored)
    }
}
