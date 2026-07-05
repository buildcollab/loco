//! # Built-in memory tools
//!
//! Framework [`Tool`]s giving an agent long-term memory (RAG): `remember`
//! persists a fact/summary, `search_memory` retrieves the most relevant
//! memories for a query. They reach the agent's
//! [`MemoryStore`](crate::agui::context::MemoryStore) through the [`ToolContext`]
//! — the framework injects a DB-backed store (scoped to the tenant/conversation)
//! in [`worker::execute`](crate::agui::worker::execute).

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::agui::context::{MemoryStore, NewMemory, ToolContext};
use crate::agui::provider::{ToolKind, ToolSpec};
use crate::agui::tool::{Tool, Tools};
use crate::{Error, Result};

/// The framework's built-in memory tools: `remember`, `search_memory`.
#[must_use]
pub fn builtin_memory_tools() -> Tools {
    Tools::new().with(Remember).with(SearchMemory)
}

fn store(ctx: &ToolContext) -> Result<std::sync::Arc<dyn MemoryStore>> {
    ctx.memory()
        .ok_or_else(|| Error::string("no memory store configured for this run"))
}

#[derive(Deserialize)]
struct RememberArgs {
    content: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    metadata: Option<Value>,
}

/// Persist a fact/summary to long-term memory for later retrieval.
struct Remember;

#[async_trait]
impl Tool for Remember {
    type Args = RememberArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "remember".to_string(),
            description: "Save a fact, preference, or summary to long-term memory so it can be \
                          recalled in later turns/conversations."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The text to remember." },
                    "kind": { "type": "string", "description": "Optional category, e.g. 'preference'." },
                    "metadata": { "type": "object" }
                },
                "required": ["content"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: RememberArgs) -> Result<Value> {
        let n = store(ctx)?
            .add(vec![NewMemory {
                content: args.content,
                kind: args.kind,
                metadata: args.metadata,
            }])
            .await?;
        Ok(json!({ "remembered": n }))
    }
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize {
    5
}

/// Retrieve the most relevant memories for a query (semantic when embeddings are
/// configured, lexical otherwise).
struct SearchMemory;

#[async_trait]
impl Tool for SearchMemory {
    type Args = SearchArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "search_memory".to_string(),
            description: "Search long-term memory for information relevant to a query. Returns \
                          ranked hits with their ids so you can cite them."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to look for." },
                    "top_k": { "type": "integer", "description": "Max results (default 5)." }
                },
                "required": ["query"]
            }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: SearchArgs) -> Result<Value> {
        let hits = store(ctx)?.search(&args.query, args.top_k).await?;
        Ok(json!({ "hits": serde_json::to_value(&hits)? }))
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::agui::context::MemoryHit;
    use crate::agui::runtime::ToolExecutor;

    #[derive(Default)]
    struct MemVec(Mutex<Vec<(String, Option<String>)>>);

    #[async_trait]
    impl MemoryStore for MemVec {
        async fn add(&self, items: Vec<NewMemory>) -> Result<usize> {
            let mut v = self.0.lock().unwrap();
            let n = items.len();
            for it in items {
                v.push((it.content, it.kind));
            }
            Ok(n)
        }
        async fn search(&self, query: &str, top_k: usize) -> Result<Vec<MemoryHit>> {
            let v = self.0.lock().unwrap();
            Ok(v.iter()
                .filter(|(c, _)| c.to_lowercase().contains(&query.to_lowercase()))
                .take(top_k)
                .enumerate()
                .map(|(i, (c, k))| MemoryHit {
                    id: format!("m{i}"),
                    content: c.clone(),
                    score: 1.0,
                    kind: k.clone(),
                    metadata: None,
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn remember_then_search() {
        let mem = Arc::new(MemVec::default());
        let ctx = ToolContext::default().with_memory(mem.clone());
        let tools = builtin_memory_tools();

        tools
            .execute(
                &ctx,
                "remember",
                json!({ "content": "The Q3 revenue was 4.2M", "kind": "fact" }),
            )
            .await
            .unwrap();
        let out = tools
            .execute(&ctx, "search_memory", json!({ "query": "revenue" }))
            .await
            .unwrap();
        let hits = out["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0]["content"].as_str().unwrap().contains("4.2M"));
    }

    #[tokio::test]
    async fn missing_store_errors() {
        let ctx = ToolContext::default();
        let err = builtin_memory_tools()
            .execute(&ctx, "search_memory", json!({ "query": "x" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no memory store configured"));
    }
}
