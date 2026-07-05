//! # Typed tools
//!
//! A thin, type-safe layer over [`ToolExecutor`](crate::agui::runtime::ToolExecutor).
//! Instead of hand-writing a `specs()` list *and* a parallel `match` in
//! `execute()` over raw [`serde_json::Value`] arguments (two sources of truth
//! that drift), you implement [`Tool`] once per tool: it declares its
//! [`ToolSpec`] and receives **deserialized, typed arguments**. Collect tools
//! into a [`Tools`] registry, which *is* a [`ToolExecutor`] — it derives
//! `specs()` from the tools and routes `execute` by name, turning a bad-argument
//! payload into a clean, model-visible error rather than a silent mismatch.
//!
//! ```ignore
//! use loco_rs::agui::{Tool, ToolContext, Tools, ToolSpec, ToolKind};
//! use serde::Deserialize;
//! use serde_json::{json, Value};
//!
//! #[derive(Deserialize)]
//! struct SaveMemo { text: String }
//!
//! struct SaveMemoTool;
//! #[async_trait::async_trait]
//! impl Tool for SaveMemoTool {
//!     type Args = SaveMemo;
//!     fn spec(&self) -> ToolSpec {
//!         ToolSpec {
//!             name: "save_memo".into(),
//!             description: "Persist a short memo.".into(),
//!             parameters: json!({
//!                 "type": "object",
//!                 "properties": { "text": { "type": "string" } },
//!                 "required": ["text"]
//!             }),
//!             kind: ToolKind::Write,
//!         }
//!     }
//!     async fn call(&self, _ctx: &ToolContext, args: SaveMemo) -> loco_rs::Result<Value> {
//!         Ok(json!({ "saved": true, "text": args.text }))
//!     }
//! }
//!
//! let tools = Tools::new().with(SaveMemoTool);
//! ```

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::agui::context::ToolContext;
use crate::agui::provider::ToolSpec;
use crate::agui::runtime::ToolExecutor;
use crate::{Error, Result};

/// A single, typed tool the model may call.
///
/// [`Args`](Tool::Args) is deserialized from the model-supplied JSON before
/// [`call`](Tool::call) runs, so implementations work with a concrete struct.
/// Use [`NoArgs`] for tools that take no parameters.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Typed arguments deserialized from the tool call payload.
    type Args: DeserializeOwned + Send;

    /// The tool's advertised spec (name, description, JSON-schema, read/write
    /// kind). This is the single source of truth — the registry derives its
    /// `specs()` and its dispatch table from it.
    fn spec(&self) -> ToolSpec;

    /// Run the tool with the run's [`ToolContext`] (app deps, principal, scope,
    /// token resolver, artifact store, custom deps) and typed, validated
    /// arguments.
    ///
    /// # Errors
    /// Tool failures surface as `Err`; the run-loop records them as an `error`
    /// tool result the model sees, and continues.
    async fn call(&self, ctx: &ToolContext, args: Self::Args) -> Result<Value>;
}

/// Arguments type for a tool that takes none. Deserializes from `{}` (and from
/// `null`, which the registry normalizes), ignoring any extra fields.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct NoArgs {}

/// Object-safe erasure of a [`Tool`] so heterogeneous tools live in one `Vec`.
#[async_trait]
trait ErasedTool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call_raw(&self, ctx: &ToolContext, args: Value) -> Result<Value>;
}

struct ToolHolder<T: Tool>(T);

#[async_trait]
impl<T: Tool> ErasedTool for ToolHolder<T> {
    fn spec(&self) -> ToolSpec {
        self.0.spec()
    }

    async fn call_raw(&self, ctx: &ToolContext, args: Value) -> Result<Value> {
        // Tools with no parameters are frequently called with `null`; treat it
        // as an empty object so `NoArgs`-style structs deserialize.
        let args = if args.is_null() { json!({}) } else { args };
        let parsed: T::Args = serde_json::from_value(args).map_err(|e| {
            Error::Message(format!(
                "invalid arguments for tool '{}': {e}",
                self.0.spec().name
            ))
        })?;
        self.0.call(ctx, parsed).await
    }
}

/// A registry of typed [`Tool`]s that implements
/// [`ToolExecutor`](crate::agui::runtime::ToolExecutor).
///
/// Pass a `Tools` straight into the run-loop where a hand-written
/// `ToolExecutor` would go. `specs()` is derived from the registered tools and
/// `execute` routes by name (unknown name → clean error).
#[derive(Default)]
pub struct Tools {
    tools: Vec<Box<dyn ErasedTool>>,
}

impl Tools {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool (builder style).
    #[must_use]
    pub fn with<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(ToolHolder(tool)));
        self
    }

    /// Register a tool in place.
    pub fn add<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        self.tools.push(Box::new(ToolHolder(tool)));
        self
    }

    /// Number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[async_trait]
impl ToolExecutor for Tools {
    fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    async fn execute(&self, ctx: &ToolContext, name: &str, args: Value) -> Result<Value> {
        for tool in &self.tools {
            if tool.spec().name == name {
                return tool.call_raw(ctx, args).await;
            }
        }
        Err(Error::Message(format!("unknown tool: {name}")))
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::provider::ToolKind;

    #[derive(Deserialize)]
    struct EchoArgs {
        text: String,
    }

    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        type Args = EchoArgs;
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "echo".to_string(),
                description: "Echo the text back".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
                kind: ToolKind::Read,
            }
        }
        async fn call(&self, _ctx: &ToolContext, args: EchoArgs) -> Result<Value> {
            Ok(json!({ "echoed": args.text }))
        }
    }

    struct PingTool;
    #[async_trait]
    impl Tool for PingTool {
        type Args = NoArgs;
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "ping".to_string(),
                description: "no-arg tool".to_string(),
                parameters: json!({ "type": "object", "properties": {} }),
                kind: ToolKind::Read,
            }
        }
        async fn call(&self, _ctx: &ToolContext, _args: NoArgs) -> Result<Value> {
            Ok(json!({ "pong": true }))
        }
    }

    #[tokio::test]
    async fn derives_specs_and_dispatches() {
        let tools = Tools::new().with(EchoTool).with(PingTool);
        let ctx = ToolContext::default();
        let names: Vec<String> = tools.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["echo".to_string(), "ping".to_string()]);

        let out = tools
            .execute(&ctx, "echo", json!({ "text": "hi" }))
            .await
            .unwrap();
        assert_eq!(out, json!({ "echoed": "hi" }));

        // no-arg tool called with null args
        let out = tools.execute(&ctx, "ping", Value::Null).await.unwrap();
        assert_eq!(out, json!({ "pong": true }));
    }

    #[tokio::test]
    async fn bad_args_are_a_clean_error() {
        let tools = Tools::new().with(EchoTool);
        let ctx = ToolContext::default();
        // Missing required `text` → deserialize error, surfaced (not a panic).
        let err = tools.execute(&ctx, "echo", json!({})).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("invalid arguments for tool 'echo'"));
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let tools = Tools::new().with(EchoTool);
        let ctx = ToolContext::default();
        let err = tools.execute(&ctx, "nope", json!({})).await.unwrap_err();
        assert!(err.to_string().contains("unknown tool"));
    }
}
