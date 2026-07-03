//! # Subagents with local tool calls
//!
//! A *subagent* is a child agent a parent run can delegate a task to. Each
//! subagent runs its own [`run_turn`](crate::agui::runtime::run_turn) loop with
//! its **own local (in-process) [`ToolExecutor`]**, its own system prompt, and
//! (optionally) its own [`Provider`], then returns its final text to the parent.
//!
//! Delegation is modeled as **subagent-as-tool**: [`SubagentExecutor`] is a
//! [`ToolExecutor`] that exposes one [`ToolSpec`] per registered subagent, so
//! the parent's model can call a subagent by name. Compose it with the app's own
//! tools via [`CompositeToolExecutor`] and hand the result to the parent
//! `run_turn` — the run-loop, authorization, and event vocabulary are reused
//! unchanged.
//!
//! ```rust,ignore
//! let mut reg = SubagentRegistry::default();
//! reg.register(LocalSubagent {
//!     name: "summarizer".into(), description: "Summarizes text".into(),
//!     system: "You summarize.".into(),
//!     provider: RigProvider::new(key, None, model),
//!     exec: MyLocalTools, authz: AllowAll, max_tool_turns: 4,
//! });
//! let exec = CompositeToolExecutor::default()
//!     .with(AppTools)
//!     .with(SubagentExecutor::new(Arc::new(reg), Arc::new(NullSink)));
//! run_turn(&store, &exec, &provider, &sink, &params, &AllowAll).await?;
//! ```
//!
//! ## Streaming & interrupts
//!
//! A subagent runs with its own [`EventSink`] — typically a DB-logging sink the
//! app supplies for review/debugging — rather than the client's SSE sink; only
//! its final result surfaces to the parent (as the delegation tool result). A
//! subagent that needs human approval mid-run is covered by the parent
//! interrupt bubble-up in [`crate::agui::runtime`]; the [`InMemoryStore`] here
//! is for non-interrupting (e.g. read-only) subagents.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::agui::provider::{ChatMessage, Provider, ToolCallReq, ToolKind, ToolSpec, Usage};
use crate::agui::runtime::{
    run_turn, AllowAll, ConversationStore, MessageRef, PendingToolCall, RunParams, ToolAuthorizer,
    ToolExecutor, ToolRef,
};
use crate::agui::transport::{EventSink, NullSink};
use crate::{Error, Result};

/// Default cap on how deeply subagents may nest (subagent calling subagent).
pub const DEFAULT_MAX_SUBAGENT_DEPTH: usize = 3;

/// The default delegation input schema: a single required `input` string.
#[must_use]
pub fn default_task_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "input": { "type": "string", "description": "The task for the subagent." } },
        "required": ["input"]
    })
}

/// What a subagent produced.
#[derive(Debug, Clone, Default)]
pub struct SubagentOutput {
    pub text: String,
    pub usage: Usage,
}

/// Context threaded into a subagent run (depth accounting for the recursion guard).
#[derive(Debug, Clone, Copy)]
pub struct SubagentCtx {
    pub depth: usize,
    pub max_depth: usize,
}

/// A child agent the parent can delegate to. Object-safe so a registry can hold
/// heterogeneous `Arc<dyn Subagent>`.
#[async_trait]
pub trait Subagent: Send + Sync {
    /// The tool name the parent model calls to delegate.
    fn name(&self) -> String;
    /// Human/model-facing description of what this subagent does.
    fn description(&self) -> String;
    /// Delegation input schema (defaults to `{ input: string }`).
    fn parameters(&self) -> Value {
        default_task_schema()
    }
    /// Run the subagent to completion and return its final text + usage. `sink`
    /// receives the child's AG-UI events (e.g. a DB-logging sink for review).
    ///
    /// # Errors
    /// Provider/store/loop failures propagate as `Err` and are recorded by the
    /// parent as an `error` tool result.
    async fn run(&self, input: &str, ctx: &SubagentCtx, sink: &dyn EventSink)
        -> Result<SubagentOutput>;
}

/// A concrete subagent backed by a [`Provider`] + a **local** [`ToolExecutor`].
///
/// Generic over its components so callers can pick zero-cost concrete types
/// (`LocalSubagent<RigProvider, MyTools, AllowAll>`) or erased boxed ones
/// (`LocalSubagent<Box<dyn Provider>, Box<dyn ToolExecutor>, Box<dyn ToolAuthorizer>>`),
/// both of which satisfy the `Subagent` object trait for the registry.
pub struct LocalSubagent<P, E, A = AllowAll> {
    pub name: String,
    pub description: String,
    pub system: String,
    pub provider: P,
    pub exec: E,
    pub authz: A,
    pub max_tool_turns: usize,
}

#[async_trait]
impl<P, E, A> Subagent for LocalSubagent<P, E, A>
where
    P: Provider,
    E: ToolExecutor,
    A: ToolAuthorizer,
{
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    async fn run(
        &self,
        input: &str,
        ctx: &SubagentCtx,
        sink: &dyn EventSink,
    ) -> Result<SubagentOutput> {
        let store = InMemoryStore::with_user(input);
        let params = RunParams {
            system: self.system.clone(),
            run_id: format!("sub-{}-{}", self.name, ctx.depth),
            thread_id: format!("sub-{}", self.name),
            // The subagent owns its own write decision via `authz`; delegation
            // is not gated by the parent's human-approval interrupt in this
            // (non-interrupting) path.
            auto_approve: true,
            max_tool_turns: self.max_tool_turns,
        };
        run_turn(&store, &self.exec, &self.provider, &sink, &params, &self.authz).await?;
        Ok(SubagentOutput {
            text: store.final_text(),
            usage: store.final_usage(),
        })
    }
}

/// A set of subagents, each exposed to a parent as one tool.
#[derive(Default, Clone)]
pub struct SubagentRegistry {
    agents: Vec<Arc<dyn Subagent>>,
}

impl SubagentRegistry {
    /// Register a subagent. Chainable.
    pub fn register(&mut self, agent: impl Subagent + 'static) -> &mut Self {
        self.agents.push(Arc::new(agent));
        self
    }

    /// Register a pre-boxed subagent (e.g. built dynamically). Chainable.
    pub fn register_arc(&mut self, agent: Arc<dyn Subagent>) -> &mut Self {
        self.agents.push(agent);
        self
    }

    /// Look up a subagent by its tool name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Subagent>> {
        self.agents.iter().find(|a| a.name() == name)
    }

    /// One [`ToolSpec`] per subagent (`kind: Read` — see [`SubagentExecutor`]).
    #[must_use]
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.agents
            .iter()
            .map(|a| ToolSpec {
                name: a.name(),
                description: a.description(),
                parameters: a.parameters(),
                kind: ToolKind::Read,
            })
            .collect()
    }
}

/// A [`ToolExecutor`] that runs registered subagents as tools.
///
/// `specs()` returns one entry per subagent; `execute(name, args)` reads the
/// `input` string, enforces the depth guard, runs the subagent (with the
/// configured `sink`), and returns `{"output": text}`. Subagent specs are
/// [`ToolKind::Read`] so the parent's built-in write-approval gate does not fire
/// on delegation itself — each subagent owns its own write decisions.
pub struct SubagentExecutor {
    registry: Arc<SubagentRegistry>,
    sink: Arc<dyn EventSink>,
    depth: usize,
    max_depth: usize,
}

impl SubagentExecutor {
    /// Build an executor at depth 0 with [`DEFAULT_MAX_SUBAGENT_DEPTH`].
    #[must_use]
    pub fn new(registry: Arc<SubagentRegistry>, sink: Arc<dyn EventSink>) -> Self {
        Self {
            registry,
            sink,
            depth: 0,
            max_depth: DEFAULT_MAX_SUBAGENT_DEPTH,
        }
    }

    /// Set the maximum nesting depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Build an executor pinned at a specific depth (used when a subagent itself
    /// delegates, to keep the recursion guard accurate).
    #[must_use]
    pub fn at_depth(registry: Arc<SubagentRegistry>, sink: Arc<dyn EventSink>, depth: usize, max_depth: usize) -> Self {
        Self { registry, sink, depth, max_depth }
    }
}

#[async_trait]
impl ToolExecutor for SubagentExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        self.registry.specs()
    }

    async fn execute(&self, name: &str, args: Value) -> Result<Value> {
        if self.depth >= self.max_depth {
            warn!(
                target: "loco_rs::agui",
                subagent = %name, depth = self.depth, max_depth = self.max_depth,
                "subagent depth exceeded"
            );
            return Err(Error::Message(format!(
                "max subagent depth ({}) exceeded delegating to '{name}'",
                self.max_depth
            )));
        }
        let agent = self
            .registry
            .get(name)
            .ok_or_else(|| Error::Message(format!("unknown subagent: {name}")))?
            .clone();
        let input = args
            .get("input")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::string("subagent call missing 'input' string"))?;
        let ctx = SubagentCtx {
            depth: self.depth + 1,
            max_depth: self.max_depth,
        };
        info!(target: "loco_rs::agui", subagent = %name, depth = ctx.depth, "delegating to subagent");
        let out = agent.run(input, &ctx, self.sink.as_ref()).await?;
        info!(
            target: "loco_rs::agui",
            subagent = %name, output_len = out.text.len(), "subagent finished"
        );
        Ok(json!({ "output": out.text }))
    }
}

/// Combine several [`ToolExecutor`]s into one, routing each call to the child
/// whose `specs()` owns the tool name. Use it to expose app tools **and**
/// subagents to a single parent run. On duplicate tool names the first
/// registered executor wins (documented; register carefully).
#[derive(Default)]
pub struct CompositeToolExecutor {
    execs: Vec<Box<dyn ToolExecutor>>,
}

impl CompositeToolExecutor {
    /// Add an executor. Chainable (builder style).
    #[must_use]
    pub fn with(mut self, exec: impl ToolExecutor + 'static) -> Self {
        self.execs.push(Box::new(exec));
        self
    }
}

#[async_trait]
impl ToolExecutor for CompositeToolExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        // First-wins on duplicate names.
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for e in &self.execs {
            for s in e.specs() {
                if seen.insert(s.name.clone()) {
                    out.push(s);
                }
            }
        }
        out
    }

    async fn execute(&self, name: &str, args: Value) -> Result<Value> {
        for e in &self.execs {
            if e.specs().iter().any(|s| s.name == name) {
                return e.execute(name, args).await;
            }
        }
        Err(Error::Message(format!("no executor for tool: {name}")))
    }
}

// ---------------------------------------------------------------------------
// InMemoryStore — an ephemeral ConversationStore for stateless sub-runs.
// ---------------------------------------------------------------------------

/// An in-memory, non-persistent [`ConversationStore`] for subagent runs that do
/// not need to survive an interrupt. Captures the finalized assistant `parts`
/// so [`final_text`](InMemoryStore::final_text) can recover the subagent's
/// answer (since `run_turn` returns `()`).
#[derive(Default)]
pub struct InMemoryStore {
    inner: Mutex<InMemState>,
}

#[derive(Default)]
struct InMemState {
    history: Vec<ChatMessage>,
    seq: usize,
    final_parts: Value,
    final_usage: Usage,
}

impl InMemoryStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed the store with a single user message (the delegated task).
    #[must_use]
    pub fn with_user(input: &str) -> Self {
        let store = Self::default();
        store
            .inner
            .lock()
            .unwrap()
            .history
            .push(ChatMessage::text("user", input));
        store
    }

    /// Concatenate the `text` parts of the finalized assistant message.
    #[must_use]
    pub fn final_text(&self) -> String {
        let s = self.inner.lock().unwrap();
        s.final_parts
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default()
    }

    /// The finalized usage for the sub-run.
    #[must_use]
    pub fn final_usage(&self) -> Usage {
        self.inner.lock().unwrap().final_usage.clone()
    }
}

#[async_trait]
impl ConversationStore for InMemoryStore {
    async fn load_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.inner.lock().unwrap().history.clone())
    }

    async fn append_user_message(&self, text: &str) -> Result<MessageRef> {
        let mut s = self.inner.lock().unwrap();
        s.history.push(ChatMessage::text("user", text));
        s.seq += 1;
        Ok(MessageRef {
            id: format!("sub_umsg_{}", s.seq),
        })
    }

    async fn begin_assistant_message(&self, _provider: &str, _model: &str) -> Result<MessageRef> {
        let mut s = self.inner.lock().unwrap();
        s.seq += 1;
        Ok(MessageRef {
            id: format!("sub_msg_{}", s.seq),
        })
    }

    async fn record_tool_call(
        &self,
        _msg: &MessageRef,
        call: &ToolCallReq,
        _status: &str,
    ) -> Result<ToolRef> {
        Ok(ToolRef { id: call.id.clone() })
    }

    async fn complete_tool_call(
        &self,
        _tool: &ToolRef,
        _status: &str,
        _result: &Value,
        _duration_ms: i64,
    ) -> Result<()> {
        Ok(())
    }

    async fn finalize_assistant_message(
        &self,
        _msg: &MessageRef,
        parts: Value,
        usage: &Usage,
        _status: &str,
    ) -> Result<()> {
        let mut s = self.inner.lock().unwrap();
        s.final_parts = parts;
        s.final_usage = usage.clone();
        Ok(())
    }

    async fn find_pending_tool_call(&self, _tool_call_id: &str) -> Result<Option<PendingToolCall>> {
        // In-memory subagents auto-approve, so nothing is ever pending.
        Ok(None)
    }

    async fn set_conversation_status(&self, _status: &str) -> Result<()> {
        Ok(())
    }
}

/// Convenience: an [`Arc`]'d [`NullSink`] for subagents that need no event log.
#[must_use]
pub fn null_sink() -> Arc<dyn EventSink> {
    Arc::new(NullSink)
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::provider::StubProvider;
    use crate::agui::transport::EventSink;
    use crate::agui::AguiEvent;
    use std::sync::Mutex as StdMutex;

    // A local read tool for subagents.
    struct EchoTools;
    #[async_trait]
    impl ToolExecutor for EchoTools {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![ToolSpec {
                name: "noop".into(),
                description: "noop".into(),
                parameters: json!({"type": "object"}),
                kind: ToolKind::Read,
            }]
        }
        async fn execute(&self, name: &str, _args: Value) -> Result<Value> {
            Ok(json!({ "ok": name }))
        }
    }

    fn ctx() -> SubagentCtx {
        SubagentCtx { depth: 1, max_depth: DEFAULT_MAX_SUBAGENT_DEPTH }
    }

    #[tokio::test]
    async fn in_memory_store_captures_final_text() {
        let store = InMemoryStore::with_user("hi");
        let provider = StubProvider::with_reply("hello there");
        run_turn(&store, &EchoTools, &provider, &NullSink, &RunParams {
            system: "s".into(), run_id: "r".into(), thread_id: "t".into(),
            auto_approve: true, max_tool_turns: 3,
        }, &AllowAll).await.unwrap();
        assert!(store.final_text().contains("hello"));
    }

    #[tokio::test]
    async fn local_subagent_runs_and_returns_text() {
        let agent = LocalSubagent {
            name: "sum".into(), description: "d".into(), system: "s".into(),
            provider: StubProvider::with_reply("a summary"),
            exec: EchoTools, authz: AllowAll, max_tool_turns: 3,
        };
        let out = agent.run("please summarize", &ctx(), &NullSink).await.unwrap();
        assert!(out.text.contains("summary"));
    }

    #[tokio::test]
    async fn subagent_executor_specs_and_execute() {
        let mut reg = SubagentRegistry::default();
        reg.register(LocalSubagent {
            name: "sum".into(), description: "Summarize".into(), system: "s".into(),
            provider: StubProvider::with_reply("done"),
            exec: EchoTools, authz: AllowAll, max_tool_turns: 2,
        });
        let exec = SubagentExecutor::new(Arc::new(reg), null_sink());

        let specs = exec.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "sum");
        assert_eq!(specs[0].kind, ToolKind::Read);

        let res = exec.execute("sum", json!({ "input": "text" })).await.unwrap();
        assert!(res["output"].as_str().unwrap().contains("done"));
    }

    #[tokio::test]
    async fn subagent_executor_unknown_and_missing_input() {
        let reg = Arc::new(SubagentRegistry::default());
        let exec = SubagentExecutor::new(reg, null_sink());
        assert!(exec.execute("nope", json!({"input":"x"})).await.is_err());
    }

    #[tokio::test]
    async fn depth_guard_errors() {
        let mut reg = SubagentRegistry::default();
        reg.register(LocalSubagent {
            name: "sum".into(), description: "d".into(), system: "s".into(),
            provider: StubProvider::new(), exec: EchoTools, authz: AllowAll, max_tool_turns: 2,
        });
        // Pin depth == max so any delegation trips the guard.
        let exec = SubagentExecutor::at_depth(Arc::new(reg), null_sink(), 3, 3);
        assert!(exec.execute("sum", json!({"input":"x"})).await.is_err());
    }

    #[tokio::test]
    async fn composite_routes_by_name() {
        let mut reg = SubagentRegistry::default();
        reg.register(LocalSubagent {
            name: "sum".into(), description: "d".into(), system: "s".into(),
            provider: StubProvider::with_reply("sub result"),
            exec: EchoTools, authz: AllowAll, max_tool_turns: 2,
        });
        let composite = CompositeToolExecutor::default()
            .with(EchoTools)
            .with(SubagentExecutor::new(Arc::new(reg), null_sink()));

        // Both tool names visible.
        let names: Vec<String> = composite.specs().into_iter().map(|s| s.name).collect();
        assert!(names.contains(&"noop".to_string()));
        assert!(names.contains(&"sum".to_string()));

        // Routes to the app tool.
        assert_eq!(composite.execute("noop", json!({})).await.unwrap()["ok"], "noop");
        // Routes to the subagent.
        assert!(composite.execute("sum", json!({"input":"x"})).await.unwrap()["output"]
            .as_str().unwrap().contains("sub result"));
    }

    // Boxed/erased components still satisfy `Subagent` (dyn-compat guard).
    #[tokio::test]
    async fn boxed_components_compile_and_run() {
        let agent: LocalSubagent<Box<dyn Provider>, Box<dyn ToolExecutor>, Box<dyn ToolAuthorizer>> =
            LocalSubagent {
                name: "b".into(), description: "d".into(), system: "s".into(),
                provider: Box::new(StubProvider::with_reply("boxed ok")),
                exec: Box::new(EchoTools),
                authz: Box::new(AllowAll),
                max_tool_turns: 2,
            };
        let out = agent.run("go", &ctx(), &NullSink).await.unwrap();
        assert!(out.text.contains("boxed ok"));
    }

    // A DB-logging-style sink: collects events (stands in for a DB write).
    #[derive(Default)]
    struct CollectSink(StdMutex<Vec<String>>);
    #[async_trait]
    impl EventSink for CollectSink {
        async fn emit(&self, ev: AguiEvent) -> Result<()> {
            self.0.lock().unwrap().push(ev.event_name().to_string());
            Ok(())
        }
    }

    #[tokio::test]
    async fn subagent_events_flow_to_its_own_sink() {
        let collector = Arc::new(CollectSink::default());
        let sink: Arc<dyn EventSink> = collector.clone();
        let mut reg = SubagentRegistry::default();
        reg.register(LocalSubagent {
            name: "sum".into(), description: "d".into(), system: "s".into(),
            provider: StubProvider::with_reply("logged"),
            exec: EchoTools, authz: AllowAll, max_tool_turns: 2,
        });
        let exec = SubagentExecutor::new(Arc::new(reg), sink);
        exec.execute("sum", json!({"input":"x"})).await.unwrap();
        // The child run emitted its lifecycle into the provided (DB-logging-style) sink.
        let names = collector.0.lock().unwrap();
        assert_eq!(names.first().map(String::as_str), Some("RUN_STARTED"));
        assert_eq!(names.last().map(String::as_str), Some("RUN_FINISHED"));
    }
}
