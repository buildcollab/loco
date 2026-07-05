//! # Built-in artifact tools
//!
//! Framework [`Tool`]s that let an agent create and manage persistent
//! [`Artifact`](crate::agui::context::Artifact)s — deliverables (documents,
//! reports) that outlive the message stream. They reach the conversation's
//! [`ArtifactStore`](crate::agui::context::ArtifactStore) through the
//! [`ToolContext`], and emit a `CUSTOM` protocol event on each change so a
//! streaming (or reconnecting, multi-node) client observes it.
//!
//! Compose them into a run with
//! [`builtin_artifact_tools`] — the framework does this in
//! [`worker::execute`](crate::agui::worker::execute) when artifacts are enabled,
//! so an app gets them without wiring each tool.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::agui::context::{ArtifactStore, NewArtifact, ToolContext};
use crate::agui::protocol::AguiEvent;
use crate::agui::provider::{ToolKind, ToolSpec};
use crate::agui::tool::{Tool, Tools};
use crate::{Error, Result};

/// The framework's built-in artifact tools:
/// `create_artifact`, `update_artifact`, `get_artifact`, `list_artifacts`,
/// `tag_artifact`.
#[must_use]
pub fn builtin_artifact_tools() -> Tools {
    Tools::new()
        .with(CreateArtifact)
        .with(UpdateArtifact)
        .with(GetArtifact)
        .with(ListArtifacts)
        .with(TagArtifact)
}

fn store(ctx: &ToolContext) -> Result<std::sync::Arc<dyn ArtifactStore>> {
    ctx.artifacts()
        .ok_or_else(|| Error::string("no artifact store configured for this run"))
}

/// Emit an artifact change as a `CUSTOM` event so live/reconnecting clients see
/// it. Best-effort: a sink error does not fail the tool.
async fn emit_change(ctx: &ToolContext, action: &str, artifact: &Value) {
    if let Some(sink) = ctx.sink() {
        let _ = sink
            .emit(AguiEvent::Custom {
                name: "artifact".to_string(),
                value: json!({ "action": action, "artifact": artifact }),
            })
            .await;
    }
}

#[derive(Deserialize)]
struct CreateArgs {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reference: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    metadata: Option<Value>,
}

/// Create a new persistent artifact for the conversation.
struct CreateArtifact;

#[async_trait]
impl Tool for CreateArtifact {
    type Args = CreateArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "create_artifact".to_string(),
            description: "Create a persistent artifact (a document/report/deliverable) for this \
                          conversation. Returns the artifact with its id."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Human-readable name." },
                    "kind": { "type": "string", "description": "Content type/kind, e.g. text/markdown." },
                    "content": { "type": "string", "description": "Inline textual content." },
                    "reference": { "type": "string", "description": "Pointer to external content (e.g. a storage key)." },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for organizing/fetching." },
                    "metadata": { "type": "object", "description": "App-defined metadata." }
                },
                "required": ["name"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: CreateArgs) -> Result<Value> {
        let artifact = store(ctx)?
            .create(NewArtifact {
                name: args.name,
                kind: args.kind,
                content: args.content,
                reference: args.reference,
                tags: args.tags,
                metadata: args.metadata,
            })
            .await?;
        let value = serde_json::to_value(&artifact)?;
        emit_change(ctx, "created", &value).await;
        Ok(value)
    }
}

#[derive(Deserialize)]
struct UpdateArgs {
    pid: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    metadata: Option<Value>,
}

/// Update an artifact's content/tags/metadata (bumps its version).
struct UpdateArtifact;

#[async_trait]
impl Tool for UpdateArtifact {
    type Args = UpdateArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "update_artifact".to_string(),
            description: "Update an existing artifact's content, tags, or metadata by its id. \
                          Unspecified fields are left unchanged."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pid": { "type": "string", "description": "The artifact id." },
                    "content": { "type": "string" },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "metadata": { "type": "object" }
                },
                "required": ["pid"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: UpdateArgs) -> Result<Value> {
        let artifact = store(ctx)?
            .update(&args.pid, args.content, args.tags, args.metadata)
            .await?;
        let value = serde_json::to_value(&artifact)?;
        emit_change(ctx, "updated", &value).await;
        Ok(value)
    }
}

#[derive(Deserialize)]
struct GetArgs {
    pid: String,
}

/// Fetch one artifact by id.
struct GetArtifact;

#[async_trait]
impl Tool for GetArtifact {
    type Args = GetArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_artifact".to_string(),
            description: "Fetch a single artifact for this conversation by its id.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "pid": { "type": "string", "description": "The artifact id." } },
                "required": ["pid"]
            }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: GetArgs) -> Result<Value> {
        match store(ctx)?.get(&args.pid).await? {
            Some(a) => Ok(serde_json::to_value(&a)?),
            None => Ok(json!({ "found": false })),
        }
    }
}

#[derive(Deserialize)]
struct ListArgs {
    #[serde(default)]
    tag: Option<String>,
}

/// List the conversation's artifacts, optionally filtered by tag.
struct ListArtifacts;

#[async_trait]
impl Tool for ListArtifacts {
    type Args = ListArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_artifacts".to_string(),
            description: "List this conversation's artifacts, optionally filtered to those with a \
                          given tag."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "tag": { "type": "string", "description": "Only artifacts carrying this tag." } }
            }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: ListArgs) -> Result<Value> {
        let items = store(ctx)?.list(args.tag.as_deref()).await?;
        Ok(json!({ "artifacts": serde_json::to_value(&items)? }))
    }
}

#[derive(Deserialize)]
struct TagArgs {
    pid: String,
    tags: Vec<String>,
}

/// Add tags to an artifact (merged with its existing tags).
struct TagArtifact;

#[async_trait]
impl Tool for TagArtifact {
    type Args = TagArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tag_artifact".to_string(),
            description: "Add one or more tags to an artifact (e.g. mark it 'published'). Tags \
                          merge with the artifact's existing tags."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pid": { "type": "string", "description": "The artifact id." },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["pid", "tags"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: TagArgs) -> Result<Value> {
        let store = store(ctx)?;
        let current = store
            .get(&args.pid)
            .await?
            .ok_or_else(|| Error::string("artifact not found"))?;
        let mut tags = current.tags;
        for t in args.tags {
            if !tags.contains(&t) {
                tags.push(t);
            }
        }
        let artifact = store.update(&args.pid, None, Some(tags), None).await?;
        let value = serde_json::to_value(&artifact)?;
        emit_change(ctx, "updated", &value).await;
        Ok(value)
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::agui::context::Artifact;
    use crate::agui::runtime::ToolExecutor;
    use crate::agui::transport::EventSink;

    /// An in-memory [`ArtifactStore`] for exercising the tools without a DB.
    #[derive(Default)]
    struct MemArtifacts(Mutex<Vec<Artifact>>);

    #[async_trait]
    impl ArtifactStore for MemArtifacts {
        async fn create(&self, new: NewArtifact) -> Result<Artifact> {
            let mut v = self.0.lock().unwrap();
            let a = Artifact {
                pid: format!("art-{}", v.len() + 1),
                name: new.name,
                kind: new.kind,
                content: new.content,
                reference: new.reference,
                tags: new.tags,
                metadata: new.metadata,
                version: 1,
            };
            v.push(a.clone());
            Ok(a)
        }
        async fn update(
            &self,
            pid: &str,
            content: Option<String>,
            tags: Option<Vec<String>>,
            metadata: Option<Value>,
        ) -> Result<Artifact> {
            let mut v = self.0.lock().unwrap();
            let a = v
                .iter_mut()
                .find(|a| a.pid == pid)
                .ok_or_else(|| Error::string("not found"))?;
            if let Some(c) = content {
                a.content = Some(c);
            }
            if let Some(t) = tags {
                a.tags = t;
            }
            if let Some(m) = metadata {
                a.metadata = Some(m);
            }
            a.version += 1;
            Ok(a.clone())
        }
        async fn get(&self, pid: &str) -> Result<Option<Artifact>> {
            Ok(self.0.lock().unwrap().iter().find(|a| a.pid == pid).cloned())
        }
        async fn list(&self, tag: Option<&str>) -> Result<Vec<Artifact>> {
            Ok(self
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|a| tag.is_none_or(|t| a.tags.iter().any(|x| x == t)))
                .cloned()
                .collect())
        }
    }

    #[derive(Default)]
    struct CollectSink(Mutex<Vec<AguiEvent>>);

    #[async_trait]
    impl EventSink for CollectSink {
        async fn emit(&self, ev: AguiEvent) -> Result<()> {
            self.0.lock().unwrap().push(ev);
            Ok(())
        }
    }

    #[tokio::test]
    async fn create_list_tag_flow_and_custom_events() {
        let store = Arc::new(MemArtifacts::default());
        let sink = Arc::new(CollectSink::default());
        let ctx = ToolContext::default()
            .with_artifacts(store.clone())
            .with_sink(sink.clone());
        let tools = builtin_artifact_tools();

        let created = tools
            .execute(
                &ctx,
                "create_artifact",
                json!({ "name": "Report", "content": "hi", "tags": ["draft"] }),
            )
            .await
            .unwrap();
        let pid = created["pid"].as_str().unwrap().to_string();
        assert_eq!(created["version"], 1);

        let listed = tools.execute(&ctx, "list_artifacts", json!({})).await.unwrap();
        assert_eq!(listed["artifacts"].as_array().unwrap().len(), 1);

        // Filtered list by a tag that does not exist yet.
        let none = tools
            .execute(&ctx, "list_artifacts", json!({ "tag": "published" }))
            .await
            .unwrap();
        assert_eq!(none["artifacts"].as_array().unwrap().len(), 0);

        let tagged = tools
            .execute(&ctx, "tag_artifact", json!({ "pid": pid, "tags": ["published"] }))
            .await
            .unwrap();
        let tags: Vec<&str> = tagged["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        assert!(tags.contains(&"draft") && tags.contains(&"published"));
        assert_eq!(tagged["version"], 2); // bumped by update

        // Both create and tag emitted a CUSTOM "artifact" event.
        let evs = sink.0.lock().unwrap();
        let artifact_events = evs
            .iter()
            .filter(|e| matches!(e, AguiEvent::Custom { name, .. } if name == "artifact"))
            .count();
        assert_eq!(artifact_events, 2);
    }

    #[tokio::test]
    async fn missing_store_is_a_clean_error() {
        // A detached context (no artifact store) → the tool errors, not panics.
        let ctx = ToolContext::default();
        let tools = builtin_artifact_tools();
        let err = tools
            .execute(&ctx, "create_artifact", json!({ "name": "x" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no artifact store configured"));
    }
}
