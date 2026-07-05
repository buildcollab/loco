{% set mod_name = name | snake_case -%}
to: src/agents/{{mod_name}}/tools.rs
skip_exists: true
---
//! Tools for the `{{mod_name}}` agent.
//!
//! Each tool is a typed [`Tool`]: it declares its [`ToolSpec`] once and receives
//! deserialized arguments in `call`. Collect them into a [`Tools`] registry —
//! `specs()` and dispatch are derived, so there is no stringly-typed `match` to
//! keep in sync.

use async_trait::async_trait;
use loco_rs::agui::{NoArgs, Tool, ToolContext, ToolKind, ToolSpec, Tools};
use loco_rs::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

/// Build this agent's tool registry.
#[must_use]
pub fn tools() -> Tools {
    Tools::new().with(GetTime).with(SaveMemo)
}

/// Example read tool: returns the current server time.
pub struct GetTime;

#[async_trait]
impl Tool for GetTime {
    type Args = NoArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_time".to_string(),
            description: "Return the current server time.".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, _ctx: &ToolContext, _args: NoArgs) -> Result<Value> {
        Ok(json!({ "time": chrono::Utc::now().to_rfc3339() }))
    }
}

/// Typed arguments for [`SaveMemo`].
#[derive(Deserialize)]
pub struct SaveMemoArgs {
    pub text: String,
}

/// Example write tool: gated by human approval unless the run auto-approves.
pub struct SaveMemo;

#[async_trait]
impl Tool for SaveMemo {
    type Args = SaveMemoArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "save_memo".to_string(),
            description: "Persist a short memo.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, _ctx: &ToolContext, args: SaveMemoArgs) -> Result<Value> {
        Ok(json!({ "saved": true, "text": args.text }))
    }
}
