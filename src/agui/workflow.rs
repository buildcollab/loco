//! # Workflow agents — deterministic multi-agent orchestration
//!
//! [`Subagent`] delegation is *model-driven* (the LLM decides who to call). For
//! a data-collation pipeline you usually want *deterministic* control flow:
//! fan work out, run steps in a fixed order, or iterate until done. These
//! combinators — [`SequentialAgent`], [`ParallelAgent`], [`LoopAgent`] — are
//! themselves [`Subagent`]s (so they compose and register like any other) but
//! run their children with explicit orchestration instead of an LLM.
//!
//! ```ignore
//! // gather (in parallel) → synthesize (sequential), as one agent
//! let pipeline = SequentialAgent::new("report", "Build the report", vec![
//!     Arc::new(ParallelAgent::new("gather", "Pull sources", vec![sales, support, crm])),
//!     Arc::new(synthesize),
//! ]);
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use crate::agui::provider::Usage;
use crate::agui::subagent::{Subagent, SubagentCtx, SubagentOutput};
use crate::agui::transport::EventSink;
use crate::Result;

/// Run child agents **in order**, piping each one's output text into the next as
/// input; the workflow's result is the last child's output. Use for a fixed
/// pipeline (gather → transform → format).
pub struct SequentialAgent {
    name: String,
    description: String,
    steps: Vec<Arc<dyn Subagent>>,
}

impl SequentialAgent {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        steps: Vec<Arc<dyn Subagent>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            steps,
        }
    }
}

#[async_trait]
impl Subagent for SequentialAgent {
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
        let mut current = input.to_string();
        let mut usage = Usage::default();
        for step in &self.steps {
            let out = step.run(&current, ctx, sink).await?;
            usage.add(&out.usage);
            current = out.text;
        }
        Ok(SubagentOutput {
            text: current,
            usage,
        })
    }
}

/// Run all child agents **concurrently** on the same input, then combine their
/// outputs into one labelled document (`## <name>\n<output>`). Use for fan-out
/// (pull several sources at once). Any child error fails the workflow.
pub struct ParallelAgent {
    name: String,
    description: String,
    branches: Vec<Arc<dyn Subagent>>,
}

impl ParallelAgent {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        branches: Vec<Arc<dyn Subagent>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            branches,
        }
    }
}

#[async_trait]
impl Subagent for ParallelAgent {
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
        let results =
            futures_util::future::join_all(self.branches.iter().map(|b| b.run(input, ctx, sink)))
                .await;
        let mut usage = Usage::default();
        let mut sections = Vec::new();
        for (branch, res) in self.branches.iter().zip(results) {
            let out = res?; // propagate the first branch error
            usage.add(&out.usage);
            sections.push(format!("## {}\n{}", branch.name(), out.text));
        }
        Ok(SubagentOutput {
            text: sections.join("\n\n"),
            usage,
        })
    }
}

/// A predicate over a loop iteration's output deciding whether to stop.
pub type StopWhen = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Run one child agent **repeatedly**, feeding its output back as the next
/// input, until `max_iters` is reached or the optional `stop` predicate returns
/// true. Use for iterative refinement (draft → critique → revise).
pub struct LoopAgent {
    name: String,
    description: String,
    body: Arc<dyn Subagent>,
    max_iters: usize,
    stop: Option<StopWhen>,
}

impl LoopAgent {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        body: Arc<dyn Subagent>,
        max_iters: usize,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            body,
            max_iters,
            stop: None,
        }
    }

    /// Stop early when `pred(output)` is true.
    #[must_use]
    pub fn until(mut self, pred: StopWhen) -> Self {
        self.stop = Some(pred);
        self
    }
}

#[async_trait]
impl Subagent for LoopAgent {
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
        let mut current = input.to_string();
        let mut usage = Usage::default();
        for _ in 0..self.max_iters.max(1) {
            let out = self.body.run(&current, ctx, sink).await?;
            usage.add(&out.usage);
            current = out.text;
            if let Some(stop) = &self.stop {
                if stop(&current) {
                    break;
                }
            }
        }
        Ok(SubagentOutput {
            text: current,
            usage,
        })
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::transport::NullSink;

    fn ctx() -> SubagentCtx {
        SubagentCtx {
            depth: 1,
            max_depth: 3,
        }
    }

    /// A deterministic test agent that appends a suffix to its input.
    struct Append(&'static str);
    #[async_trait]
    impl Subagent for Append {
        fn name(&self) -> String {
            format!("append{}", self.0)
        }
        fn description(&self) -> String {
            "append".into()
        }
        async fn run(
            &self,
            input: &str,
            _ctx: &SubagentCtx,
            _sink: &dyn EventSink,
        ) -> Result<SubagentOutput> {
            Ok(SubagentOutput {
                text: format!("{input}{}", self.0),
                usage: Usage::default(),
            })
        }
    }

    #[tokio::test]
    async fn sequential_pipes_output_to_input() {
        let seq = SequentialAgent::new(
            "seq",
            "d",
            vec![Arc::new(Append("-a")), Arc::new(Append("-b"))],
        );
        let out = seq.run("x", &ctx(), &NullSink).await.unwrap();
        assert_eq!(out.text, "x-a-b");
    }

    #[tokio::test]
    async fn parallel_combines_labelled_sections() {
        let par = ParallelAgent::new(
            "par",
            "d",
            vec![Arc::new(Append("-1")), Arc::new(Append("-2"))],
        );
        let out = par.run("q", &ctx(), &NullSink).await.unwrap();
        assert!(out.text.contains("## append-1\nq-1"));
        assert!(out.text.contains("## append-2\nq-2"));
    }

    #[tokio::test]
    async fn loop_iterates_until_predicate() {
        let lp = LoopAgent::new("lp", "d", Arc::new(Append("!")), 10)
            .until(Arc::new(|s: &str| s.matches('!').count() >= 3));
        let out = lp.run("go", &ctx(), &NullSink).await.unwrap();
        assert_eq!(out.text, "go!!!");
    }

    #[tokio::test]
    async fn loop_respects_max_iters() {
        let lp = LoopAgent::new("lp", "d", Arc::new(Append("!")), 2);
        let out = lp.run("go", &ctx(), &NullSink).await.unwrap();
        assert_eq!(out.text, "go!!");
    }
}
