//! # Built-in interaction tools
//!
//! Richer human-in-the-loop than the built-in approve/deny gate:
//!
//! - `ask_user` pauses the run to ask the user a clarifying question. It is a
//!   write tool, so (without `auto_approve`) it raises the standard AG-UI
//!   interrupt carrying the question; the client resumes with the answer in
//!   [`ResumePayload::input`](crate::agui::protocol::ResumePayload), which the
//!   run-loop returns as the tool result (see [`ASK_USER_TOOL`]).
//! - `suggest_followups` streams suggested next questions to the UI as a `CUSTOM`
//!   "suggestions" event.
//!
//! Composed into a run by [`worker::execute`](crate::agui::worker::execute) via
//! [`builtin_interact_tools`].

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::agui::context::ToolContext;
use crate::agui::provider::{ToolKind, ToolSpec};
use crate::agui::runtime::ASK_USER_TOOL;
use crate::agui::tool::{Tool, Tools};
use crate::Result;

/// The framework's built-in interaction tools: `ask_user`, `suggest_followups`.
#[must_use]
pub fn builtin_interact_tools() -> Tools {
    Tools::new().with(AskUser).with(SuggestFollowups)
}

#[derive(Deserialize)]
struct AskArgs {
    question: String,
}

/// Ask the user a clarifying question and wait for their answer.
struct AskUser;

#[async_trait]
impl Tool for AskUser {
    type Args = AskArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ASK_USER_TOOL.to_string(),
            description: "Ask the user a clarifying question and pause for their answer. Use when \
                          you genuinely need input to proceed."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "question": { "type": "string", "description": "The question to ask." } },
                "required": ["question"]
            }),
            // Write → raises a human interrupt (unless the run auto-approves); the
            // user's answer is delivered on resume as the tool result.
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, _ctx: &ToolContext, args: AskArgs) -> Result<Value> {
        // Only reached when the run auto-approves (e.g. a subagent), i.e. there is
        // no interactive user; surface the question so the caller can handle it.
        Ok(json!({ "question": args.question, "answer": Value::Null }))
    }
}

#[derive(Deserialize)]
struct SuggestArgs {
    suggestions: Vec<String>,
}

/// Offer suggested follow-up questions/actions to the user.
struct SuggestFollowups;

#[async_trait]
impl Tool for SuggestFollowups {
    type Args = SuggestArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "suggest_followups".to_string(),
            description: "Offer the user a few suggested follow-up questions or next actions. \
                          Streams them to the UI."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "suggestions": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "2-4 short follow-up prompts."
                    }
                },
                "required": ["suggestions"]
            }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: SuggestArgs) -> Result<Value> {
        ctx.emit("suggestions", json!({ "suggestions": args.suggestions }))
            .await;
        Ok(json!({ "ok": true, "count": args.suggestions.len() }))
    }
}
