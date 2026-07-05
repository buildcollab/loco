//! # Guardrails & budgets
//!
//! Two run-loop safety seams for production agents:
//!
//! - [`Guardrail`] can **inspect and rewrite** the model input (system prompt +
//!   history) before each turn and the assistant's output text after it, or
//!   **block** the run by returning `Err`. Use it for PII redaction, prompt-
//!   injection defense, and output moderation — things [`AgentHooks`] (observe
//!   only) and [`ToolAuthorizer`] (tool gating) cannot do.
//! - [`BudgetLimiter`] is consulted before each provider turn with the run's
//!   accumulated [`Usage`] and tenancy scope, so an app can cap spend per
//!   tenant/run and stop cleanly.
//!
//! Both default to no-ops ([`NoGuardrail`], [`Unlimited`]) and are supplied
//! per-agent via [`Agent::guardrail`](crate::agui::agent::Agent::guardrail) /
//! [`Agent::budget`](crate::agui::agent::Agent::budget).

use async_trait::async_trait;
use serde_json::Value;

use crate::agui::provider::{ChatMessage, Usage};
use crate::Result;

/// Inspect/rewrite model I/O around each turn, or block the run.
#[async_trait]
pub trait Guardrail: Send + Sync {
    /// Before a provider turn: mutate `system` / `history` in place (e.g. redact
    /// PII, strip injected instructions). Return `Err` to abort the run.
    async fn on_input(&self, _system: &mut String, _history: &mut Vec<ChatMessage>) -> Result<()> {
        Ok(())
    }

    /// After a turn produces a final answer: mutate `text` in place (e.g.
    /// moderate/redact). Return `Err` to abort the run.
    async fn on_output(&self, _text: &mut String) -> Result<()> {
        Ok(())
    }
}

/// A no-op [`Guardrail`] — the default.
pub struct NoGuardrail;

#[async_trait]
impl Guardrail for NoGuardrail {}

/// Per-turn budget enforcement, keyed on the run's tenancy scope + usage.
#[async_trait]
pub trait BudgetLimiter: Send + Sync {
    /// Called before each provider turn with cumulative `usage` and the run's
    /// `scope`. Return `Err` to stop the run (surfaced as a run error).
    async fn check(&self, _scope: Option<&Value>, _usage: &Usage) -> Result<()> {
        Ok(())
    }
}

/// A [`BudgetLimiter`] that never blocks — the default.
pub struct Unlimited;

#[async_trait]
impl BudgetLimiter for Unlimited {}

/// A simple [`BudgetLimiter`] that caps total tokens (input + output) per run.
pub struct TokenBudget {
    pub max_total_tokens: i64,
}

#[async_trait]
impl BudgetLimiter for TokenBudget {
    async fn check(&self, _scope: Option<&Value>, usage: &Usage) -> Result<()> {
        let total = usage.input_tokens + usage.output_tokens;
        if total >= self.max_total_tokens {
            return Err(crate::Error::Message(format!(
                "token budget exceeded: {total} >= {}",
                self.max_total_tokens
            )));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;

    struct RedactSecrets;
    #[async_trait]
    impl Guardrail for RedactSecrets {
        async fn on_input(
            &self,
            system: &mut String,
            history: &mut Vec<ChatMessage>,
        ) -> Result<()> {
            *system = system.replace("secret", "[redacted]");
            for m in history.iter_mut() {
                m.content = m.content.replace("secret", "[redacted]");
            }
            Ok(())
        }
        async fn on_output(&self, text: &mut String) -> Result<()> {
            *text = text.replace("secret", "[redacted]");
            Ok(())
        }
    }

    #[tokio::test]
    async fn guardrail_rewrites_input_and_output() {
        let g = RedactSecrets;
        let mut system = "the secret is 42".to_string();
        let mut history = vec![ChatMessage::text("user", "tell me the secret")];
        g.on_input(&mut system, &mut history).await.unwrap();
        assert_eq!(system, "the [redacted] is 42");
        assert!(history[0].content.contains("[redacted]"));

        let mut out = "the secret is 42".to_string();
        g.on_output(&mut out).await.unwrap();
        assert_eq!(out, "the [redacted] is 42");
    }

    #[tokio::test]
    async fn token_budget_blocks_over_limit() {
        let b = TokenBudget {
            max_total_tokens: 100,
        };
        let over = Usage {
            input_tokens: 60,
            output_tokens: 60,
            cached_tokens: 0,
        };
        let under = Usage {
            input_tokens: 10,
            output_tokens: 10,
            cached_tokens: 0,
        };
        assert!(b.check(None, &over).await.is_err());
        assert!(b.check(None, &under).await.is_ok());
    }
}
