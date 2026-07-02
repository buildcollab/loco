//! # AG-UI wire protocol types
//!
//! Hand-rolled Rust types for the [AG-UI](https://docs.ag-ui.com) event
//! protocol (there is no official Rust SDK). Events are serialized with an
//! internally-tagged `type` discriminator and **camelCase** field names so the
//! JSON matches what AG-UI frontends expect on the wire.
//!
//! These types are intentionally free of any application concepts — they are
//! the transport vocabulary only. The [`crate::agui::runtime`] run-loop emits
//! them; the [`crate::agui::transport`] SSE layer serializes them.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A single AG-UI protocol event, streamed server-to-client.
///
/// Serialized with `#[serde(tag = "type")]`; each variant renders its `type`
/// as the SCREAMING_SNAKE_CASE name below and its payload fields as camelCase.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AguiEvent {
    /// A run has begun for `thread_id` / `run_id`.
    #[serde(rename = "RUN_STARTED", rename_all = "camelCase")]
    RunStarted { thread_id: String, run_id: String },

    /// A new assistant (or other role) message has started streaming.
    #[serde(rename = "TEXT_MESSAGE_START", rename_all = "camelCase")]
    TextMessageStart { message_id: String, role: String },

    /// An incremental text delta for an in-flight message.
    #[serde(rename = "TEXT_MESSAGE_CONTENT", rename_all = "camelCase")]
    TextMessageContent { message_id: String, delta: String },

    /// The in-flight message is complete.
    #[serde(rename = "TEXT_MESSAGE_END", rename_all = "camelCase")]
    TextMessageEnd { message_id: String },

    /// A tool call has started; `parent_message_id` links it to the assistant
    /// message that produced it.
    #[serde(rename = "TOOL_CALL_START", rename_all = "camelCase")]
    ToolCallStart {
        tool_call_id: String,
        tool_call_name: String,
        parent_message_id: String,
    },

    /// Incremental (JSON) argument delta for an in-flight tool call.
    #[serde(rename = "TOOL_CALL_ARGS", rename_all = "camelCase")]
    ToolCallArgs { tool_call_id: String, delta: String },

    /// The tool call's arguments are complete.
    #[serde(rename = "TOOL_CALL_END", rename_all = "camelCase")]
    ToolCallEnd { tool_call_id: String },

    /// The result of executing a tool call.
    #[serde(rename = "TOOL_CALL_RESULT", rename_all = "camelCase")]
    ToolCallResult {
        message_id: String,
        tool_call_id: String,
        content: Value,
    },

    /// The run finished. `outcome` distinguishes a normal completion from an
    /// interrupt (e.g. a human-approval gate), in which case `interrupt`
    /// carries the details the frontend needs to resume.
    #[serde(rename = "RUN_FINISHED", rename_all = "camelCase")]
    RunFinished {
        thread_id: String,
        run_id: String,
        outcome: RunOutcome,
        #[serde(skip_serializing_if = "Option::is_none")]
        interrupt: Option<Interrupt>,
    },

    /// The run errored. `message` is human-readable; `code` is optional.
    #[serde(rename = "RUN_ERROR", rename_all = "camelCase")]
    RunError {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },
}

/// Terminal outcome of a run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    /// The run completed normally.
    Success,
    /// The run paused awaiting external input (e.g. approval).
    Interrupt,
}

/// Details of an interrupt that paused a run. The frontend echoes `id` back in
/// a [`ResumeItem`] to resume.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Interrupt {
    pub id: String,
    pub reason: String,
    pub payload: Value,
}

/// Input body posted by the frontend to start (or resume) a run.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunAgentInput {
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub resume: Vec<ResumeItem>,
}

/// A single resume instruction, answering a prior [`Interrupt`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeItem {
    pub interrupt_id: String,
    pub payload: ResumePayload,
}

/// Payload for a [`ResumeItem`] — currently a simple approve/deny gate.
#[derive(Debug, Clone, Deserialize)]
pub struct ResumePayload {
    pub approved: bool,
}

impl AguiEvent {
    /// The SSE `event:` name for this event (matches the serialized `type`).
    #[must_use]
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::RunStarted { .. } => "RUN_STARTED",
            Self::TextMessageStart { .. } => "TEXT_MESSAGE_START",
            Self::TextMessageContent { .. } => "TEXT_MESSAGE_CONTENT",
            Self::TextMessageEnd { .. } => "TEXT_MESSAGE_END",
            Self::ToolCallStart { .. } => "TOOL_CALL_START",
            Self::ToolCallArgs { .. } => "TOOL_CALL_ARGS",
            Self::ToolCallEnd { .. } => "TOOL_CALL_END",
            Self::ToolCallResult { .. } => "TOOL_CALL_RESULT",
            Self::RunFinished { .. } => "RUN_FINISHED",
            Self::RunError { .. } => "RUN_ERROR",
        }
    }
}

/// Canonical "text" message part.
#[must_use]
pub fn part_text(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

/// Canonical "tool_use" message part.
#[must_use]
pub fn part_tool_use(tool_call_id: &str, name: &str, input: &Value) -> Value {
    json!({
        "type": "tool_use",
        "toolCallId": tool_call_id,
        "name": name,
        "input": input,
    })
}

/// Canonical "tool_result" message part.
#[must_use]
pub fn part_tool_result(tool_call_id: &str, status: &str, content: &Value) -> Value {
    json!({
        "type": "tool_result",
        "toolCallId": tool_call_id,
        "status": status,
        "content": content,
    })
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;

    fn ser(ev: &AguiEvent) -> Value {
        serde_json::to_value(ev).unwrap()
    }

    #[test]
    fn run_started_wire_shape() {
        let v = ser(&AguiEvent::RunStarted {
            thread_id: "t1".into(),
            run_id: "r1".into(),
        });
        assert_eq!(v["type"], "RUN_STARTED");
        assert_eq!(v["threadId"], "t1");
        assert_eq!(v["runId"], "r1");
    }

    #[test]
    fn text_message_events_wire_shape() {
        assert_eq!(
            ser(&AguiEvent::TextMessageStart {
                message_id: "m1".into(),
                role: "assistant".into()
            })["type"],
            "TEXT_MESSAGE_START"
        );
        let c = ser(&AguiEvent::TextMessageContent {
            message_id: "m1".into(),
            delta: "hi".into(),
        });
        assert_eq!(c["type"], "TEXT_MESSAGE_CONTENT");
        assert_eq!(c["messageId"], "m1");
        assert_eq!(c["delta"], "hi");
        assert_eq!(
            ser(&AguiEvent::TextMessageEnd {
                message_id: "m1".into()
            })["type"],
            "TEXT_MESSAGE_END"
        );
    }

    #[test]
    fn tool_call_events_wire_shape() {
        let s = ser(&AguiEvent::ToolCallStart {
            tool_call_id: "c1".into(),
            tool_call_name: "search".into(),
            parent_message_id: "m1".into(),
        });
        assert_eq!(s["type"], "TOOL_CALL_START");
        assert_eq!(s["toolCallId"], "c1");
        assert_eq!(s["toolCallName"], "search");
        assert_eq!(s["parentMessageId"], "m1");

        let a = ser(&AguiEvent::ToolCallArgs {
            tool_call_id: "c1".into(),
            delta: "{}".into(),
        });
        assert_eq!(a["type"], "TOOL_CALL_ARGS");
        assert_eq!(a["toolCallId"], "c1");

        let r = ser(&AguiEvent::ToolCallResult {
            message_id: "m1".into(),
            tool_call_id: "c1".into(),
            content: json!({"ok": true}),
        });
        assert_eq!(r["type"], "TOOL_CALL_RESULT");
        assert_eq!(r["content"]["ok"], true);
    }

    #[test]
    fn run_finished_success_omits_interrupt() {
        let v = ser(&AguiEvent::RunFinished {
            thread_id: "t1".into(),
            run_id: "r1".into(),
            outcome: RunOutcome::Success,
            interrupt: None,
        });
        assert_eq!(v["type"], "RUN_FINISHED");
        assert_eq!(v["outcome"], "success");
        assert!(v.get("interrupt").is_none());
    }

    #[test]
    fn run_finished_interrupt_shape() {
        let v = ser(&AguiEvent::RunFinished {
            thread_id: "t1".into(),
            run_id: "r1".into(),
            outcome: RunOutcome::Interrupt,
            interrupt: Some(Interrupt {
                id: "c1".into(),
                reason: "human_approval".into(),
                payload: json!({"name": "write_file"}),
            }),
        });
        assert_eq!(v["outcome"], "interrupt");
        assert_eq!(v["interrupt"]["id"], "c1");
        assert_eq!(v["interrupt"]["reason"], "human_approval");
    }

    #[test]
    fn run_error_shape_and_event_names() {
        let v = ser(&AguiEvent::RunError {
            message: "boom".into(),
            code: None,
        });
        assert_eq!(v["type"], "RUN_ERROR");
        assert!(v.get("code").is_none());

        let ev = AguiEvent::RunStarted {
            thread_id: "t".into(),
            run_id: "r".into(),
        };
        assert_eq!(ev.event_name(), "RUN_STARTED");
    }

    #[test]
    fn deserialize_run_agent_input() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "runId": "r1",
            "message": "hello",
        }))
        .unwrap();
        assert_eq!(input.run_id.as_deref(), Some("r1"));
        assert_eq!(input.message.as_deref(), Some("hello"));
        assert!(input.resume.is_empty());
    }

    #[test]
    fn deserialize_resume() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "resume": [{"interruptId": "c1", "payload": {"approved": true}}],
        }))
        .unwrap();
        assert_eq!(input.resume.len(), 1);
        assert_eq!(input.resume[0].interrupt_id, "c1");
        assert!(input.resume[0].payload.approved);
    }

    #[test]
    fn part_builders() {
        assert_eq!(part_text("hi")["type"], "text");
        let tu = part_tool_use("c1", "search", &json!({"q": "x"}));
        assert_eq!(tu["type"], "tool_use");
        assert_eq!(tu["toolCallId"], "c1");
        assert_eq!(tu["name"], "search");
        let tr = part_tool_result("c1", "success", &json!({"n": 1}));
        assert_eq!(tr["type"], "tool_result");
        assert_eq!(tr["status"], "success");
    }
}
