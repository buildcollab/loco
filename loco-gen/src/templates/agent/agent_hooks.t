{% set mod_name = name | snake_case -%}
{% set struct_name = name | pascal_case -%}
to: src/agents/{{mod_name}}/hooks.rs
skip_exists: true
---
//! Lifecycle hooks for the `{{mod_name}}` agent.
//!
//! Hooks are observation / side-effect insertion points around the run, each
//! LLM turn, and each tool call — the same pattern as the OpenAI Agents SDK.
//! They are NOT the security seam: authorization (deny / require-approval) runs
//! first; `before_tool` fires only for calls that were allowed.
//!
//! Every method defaults to a no-op — override the ones you need (metrics,
//! auditing, redaction, cost tracking, ...). A hook returning `Err` aborts the
//! run and surfaces as `RUN_ERROR`.

use async_trait::async_trait;
use loco_rs::agui::{AgentHooks, RunCtx, ToolCallReq, TurnOutcome};
use loco_rs::prelude::*;
use serde_json::Value;

/// Lifecycle callbacks for the `{{mod_name}}` agent.
pub struct {{struct_name}}Hooks;

#[async_trait]
impl AgentHooks for {{struct_name}}Hooks {
    async fn on_run_start(&self, ctx: &RunCtx) -> Result<()> {
        tracing::debug!(agent = %ctx.agent, run_id = %ctx.run_id, "run start");
        Ok(())
    }

    async fn after_message(&self, ctx: &RunCtx, _outcome: &TurnOutcome) -> Result<()> {
        tracing::debug!(agent = %ctx.agent, "turn complete");
        Ok(())
    }

    async fn before_tool(&self, ctx: &RunCtx, call: &ToolCallReq) -> Result<()> {
        tracing::debug!(agent = %ctx.agent, tool = %call.name, "before tool");
        Ok(())
    }

    async fn after_tool(&self, ctx: &RunCtx, call: &ToolCallReq, _result: &Value) -> Result<()> {
        tracing::debug!(agent = %ctx.agent, tool = %call.name, "after tool");
        Ok(())
    }
}
