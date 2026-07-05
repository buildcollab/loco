//! # Lightweight trajectory eval
//!
//! A tiny regression harness for agents: run an input through a
//! [`Provider`] + [`ToolExecutor`] and assert on the *trajectory* — which tools
//! were called and what the final answer contains. Drive it with a
//! [`StubProvider`](crate::agui::provider::StubProvider) for deterministic unit
//! tests, or a real provider for smoke/regression suites.
//!
//! ```ignore
//! let cases = vec![EvalCase::new("greets", "hi").expect_output_contains("hello")];
//! let report = run_suite(&provider, exec, &cases).await?;
//! assert!(report.iter().all(|o| o.passed));
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::agui::protocol::AguiEvent;
use crate::agui::provider::Provider;
use crate::agui::runtime::{run_turn, AllowAll, RunParams, ToolExecutor};
use crate::agui::subagent::InMemoryStore;
use crate::agui::transport::EventSink;
use crate::Result;

/// One evaluation case: an input plus expectations on the run's trajectory.
#[derive(Debug, Clone, Default)]
pub struct EvalCase {
    pub name: String,
    pub system: String,
    pub input: String,
    /// Tool names that must be called during the run.
    pub expect_tools: Vec<String>,
    /// Substrings the final answer must contain.
    pub expect_output_contains: Vec<String>,
}

impl EvalCase {
    #[must_use]
    pub fn new(name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: input.into(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into();
        self
    }

    #[must_use]
    pub fn expect_tool(mut self, name: impl Into<String>) -> Self {
        self.expect_tools.push(name.into());
        self
    }

    #[must_use]
    pub fn expect_output_contains(mut self, s: impl Into<String>) -> Self {
        self.expect_output_contains.push(s.into());
        self
    }
}

/// The result of evaluating one [`EvalCase`].
#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub name: String,
    pub passed: bool,
    pub output: String,
    pub tools_called: Vec<String>,
    /// Human-readable descriptions of each unmet expectation.
    pub failures: Vec<String>,
}

/// Collects the trajectory (tool-call names + streamed text) from a run.
#[derive(Default)]
struct TrajectorySink {
    tools: Mutex<Vec<String>>,
    text: Mutex<String>,
}

#[async_trait]
impl EventSink for TrajectorySink {
    async fn emit(&self, ev: AguiEvent) -> Result<()> {
        match ev {
            AguiEvent::ToolCallStart { tool_call_name, .. } => {
                self.tools.lock().unwrap().push(tool_call_name);
            }
            AguiEvent::TextMessageContent { delta, .. } => {
                self.text.lock().unwrap().push_str(&delta);
            }
            _ => {}
        }
        Ok(())
    }
}

/// Run a single case and score it against its expectations.
///
/// # Errors
/// Propagates a provider/run-loop error.
pub async fn run_case<P, E>(provider: &P, exec: Arc<E>, case: &EvalCase) -> Result<EvalOutcome>
where
    P: Provider,
    E: ToolExecutor + 'static,
{
    let store = InMemoryStore::with_user(&case.input);
    let sink = TrajectorySink::default();
    let params = RunParams {
        system: case.system.clone(),
        auto_approve: true,
        max_tool_turns: 6,
        ..Default::default()
    };
    run_turn(&store, exec, provider, &sink, &params, &AllowAll).await?;

    let tools = sink.tools.lock().unwrap().clone();
    let mut output = sink.text.lock().unwrap().clone();
    if output.is_empty() {
        output = store.final_text();
    }

    let mut failures = Vec::new();
    for t in &case.expect_tools {
        if !tools.contains(t) {
            failures.push(format!("expected tool '{t}' was not called"));
        }
    }
    for s in &case.expect_output_contains {
        if !output.contains(s) {
            failures.push(format!("output did not contain '{s}'"));
        }
    }

    Ok(EvalOutcome {
        name: case.name.clone(),
        passed: failures.is_empty(),
        output,
        tools_called: tools,
        failures,
    })
}

/// Run a suite of cases, returning one [`EvalOutcome`] per case.
///
/// # Errors
/// Propagates a provider/run-loop error from any case.
pub async fn run_suite<P, E>(
    provider: &P,
    exec: Arc<E>,
    cases: &[EvalCase],
) -> Result<Vec<EvalOutcome>>
where
    P: Provider,
    E: ToolExecutor + 'static,
{
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        out.push(run_case(provider, Arc::clone(&exec), case).await?);
    }
    Ok(out)
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::provider::{StubProvider, ToolKind, ToolSpec};
    use serde_json::{json, Value};

    struct NoTools;
    #[async_trait]
    impl ToolExecutor for NoTools {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![]
        }
        async fn execute(
            &self,
            _ctx: &crate::agui::context::ToolContext,
            _name: &str,
            _args: Value,
        ) -> Result<Value> {
            Ok(json!({}))
        }
    }

    #[tokio::test]
    async fn passing_and_failing_cases() {
        let provider = StubProvider::with_reply("the answer is 42");
        let exec = Arc::new(NoTools);

        let pass = EvalCase::new("has-42", "what is it?").expect_output_contains("42");
        let fail = EvalCase::new("wants-tool", "go").expect_tool("nonexistent");

        let report = run_suite(&provider, exec, &[pass, fail]).await.unwrap();
        assert!(report[0].passed, "case 0 should pass: {:?}", report[0]);
        assert!(!report[1].passed);
        assert!(report[1].failures[0].contains("nonexistent"));
    }
}
