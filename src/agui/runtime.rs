//! # Reusable agent run-loop
//!
//! [`run_turn`] and [`resume`] drive a streaming agent turn end to end, wiring a
//! [`Provider`] to an app-supplied [`ConversationStore`] (persistence) and
//! [`ToolExecutor`] (tool implementations), emitting [`AguiEvent`]s into an
//! [`EventSink`] as it goes.
//!
//! The loop is deliberately free of any application concepts (no citation
//! parsing, personas, scopes, or concrete tables). Everything app-specific
//! arrives through the two traits below; the app does its own post-processing
//! inside its [`ConversationStore::finalize_assistant_message`] implementation.
//!
//! ## Emitted event sequence
//!
//! ```text
//! RUN_STARTED
//! TEXT_MESSAGE_START
//!   (TEXT_MESSAGE_CONTENT*)
//!   (TOOL_CALL_START TOOL_CALL_ARGS TOOL_CALL_END TOOL_CALL_RESULT)*   // per executed tool
//! TEXT_MESSAGE_END
//! RUN_FINISHED(success)
//! ```
//!
//! A write tool without `auto_approve` short-circuits to
//! `RUN_FINISHED(interrupt)` and leaves a `pending` tool call for [`resume`].

use std::time::Instant;

use serde_json::{json, Value};

use crate::agui::protocol::{
    part_text, part_tool_result, part_tool_use, AguiEvent, Interrupt, ResumeItem, RunOutcome,
};
use crate::agui::provider::{
    ChatMessage, Provider, ToolCallReq, ToolKind, ToolSpec, TurnOutcome, Usage,
};
use crate::agui::transport::EventSink;
use crate::{Error, Result};

/// Handle to a persisted message, identified by its public id (used verbatim in
/// emitted events).
#[derive(Debug, Clone)]
pub struct MessageRef {
    pub id: String,
}

/// Handle to a persisted tool-call record.
#[derive(Debug, Clone)]
pub struct ToolRef {
    pub id: String,
}

/// A tool call awaiting human approval, recovered by [`resume`].
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: Value,
    pub message_id: String,
}

/// App-supplied tool implementations.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    /// The tools available this run (name, schema, read/write kind).
    fn specs(&self) -> Vec<ToolSpec>;

    /// Execute a tool by name.
    ///
    /// # Errors
    /// Tool failures surface as `Err`; the run-loop records them as an `error`
    /// tool result and continues.
    async fn execute(&self, name: &str, args: Value) -> Result<Value>;
}

/// App-supplied persistence for a single conversation. All ids returned here
/// are public-id strings echoed back in emitted events.
#[async_trait::async_trait]
pub trait ConversationStore: Send + Sync {
    /// Load the prior conversation history the provider should see.
    async fn load_history(&self) -> Result<Vec<ChatMessage>>;

    /// Persist a user message (called by the app before [`run_turn`]).
    async fn append_user_message(&self, text: &str) -> Result<MessageRef>;

    /// Create the assistant message this run will stream into.
    async fn begin_assistant_message(&self, provider: &str, model: &str) -> Result<MessageRef>;

    /// Record a tool call in the given status (e.g. `"pending"`).
    async fn record_tool_call(
        &self,
        msg: &MessageRef,
        call: &ToolCallReq,
        status: &str,
    ) -> Result<ToolRef>;

    /// Mark a recorded tool call complete with its result.
    async fn complete_tool_call(
        &self,
        tool: &ToolRef,
        status: &str,
        result: &Value,
        duration_ms: i64,
    ) -> Result<()>;

    /// Finalize the assistant message with its assembled `parts` blob, usage,
    /// and terminal status (`"complete"`, `"streaming"`, `"errored"`).
    async fn finalize_assistant_message(
        &self,
        msg: &MessageRef,
        parts: Value,
        usage: &Usage,
        status: &str,
    ) -> Result<()>;

    /// Look up a `pending` tool call by its id (the interrupt id).
    async fn find_pending_tool_call(&self, tool_call_id: &str) -> Result<Option<PendingToolCall>>;

    /// Set the conversation's status (`"idle"`, `"responding"`, `"errored"`).
    async fn set_conversation_status(&self, status: &str) -> Result<()>;
}

/// Parameters for a run.
#[derive(Debug, Clone)]
pub struct RunParams {
    pub system: String,
    pub run_id: String,
    pub thread_id: String,
    /// When `true`, write tools execute without an approval interrupt.
    pub auto_approve: bool,
    /// Maximum provider turns (streaming rounds) before the loop stops.
    pub max_tool_turns: usize,
}

/// Outcome of the inner streaming loop.
enum LoopResult {
    /// The model produced a final answer (or the turn budget was exhausted).
    Completed,
    /// The loop paused for approval and already finalized the message + status.
    Interrupted,
}

/// Start a fresh agent turn. The app is expected to have persisted the user's
/// message already (so it appears in [`ConversationStore::load_history`]).
///
/// # Errors
/// Propagates provider/store/sink errors. On error, best-effort emits
/// `RUN_ERROR`, sets status `errored`, and finalizes the message as `errored`.
pub async fn run_turn<S, E, P, K>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
{
    sink.emit(AguiEvent::RunStarted {
        thread_id: params.thread_id.clone(),
        run_id: params.run_id.clone(),
    })
    .await?;

    let msg = store
        .begin_assistant_message(&provider.provider_name(), &provider.model_id())
        .await?;
    sink.emit(AguiEvent::TextMessageStart {
        message_id: msg.id.clone(),
        role: "assistant".to_string(),
    })
    .await?;

    let mut history = store.load_history().await?;
    let specs = exec.specs();
    let mut parts: Vec<Value> = Vec::new();
    let mut total_usage = Usage::default();

    let result = run_loop(
        store,
        exec,
        provider,
        sink,
        params,
        &msg,
        &mut history,
        &mut parts,
        &mut total_usage,
        &specs,
    )
    .await;

    finalize_run(store, sink, params, &msg, &parts, &total_usage, result).await
}

/// Resume a previously interrupted run by answering its approval gate.
///
/// # Errors
/// Errors if no `pending` tool call matches `item.interrupt_id`, or on
/// provider/store/sink failure (same error handling as [`run_turn`]).
pub async fn resume<S, E, P, K>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    item: &ResumeItem,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
{
    let pending = store
        .find_pending_tool_call(&item.interrupt_id)
        .await?
        .ok_or_else(|| Error::string("no pending tool call found for interrupt id"))?;

    let msg = MessageRef {
        id: pending.message_id.clone(),
    };

    sink.emit(AguiEvent::RunStarted {
        thread_id: params.thread_id.clone(),
        run_id: params.run_id.clone(),
    })
    .await?;
    sink.emit(AguiEvent::TextMessageStart {
        message_id: msg.id.clone(),
        role: "assistant".to_string(),
    })
    .await?;

    let mut history = store.load_history().await?;
    let specs = exec.specs();
    let mut parts: Vec<Value> = Vec::new();
    let mut total_usage = Usage::default();

    let call = ToolCallReq {
        id: pending.tool_call_id.clone(),
        name: pending.name.clone(),
        arguments: pending.arguments.clone(),
    };
    let tref = ToolRef {
        id: pending.tool_call_id.clone(),
    };

    let result: Result<LoopResult> = async {
        if item.payload.approved {
            parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
            let (status, result) = execute_and_record(exec, store, &tref, &call).await?;
            sink.emit(AguiEvent::ToolCallResult {
                message_id: msg.id.clone(),
                tool_call_id: call.id.clone(),
                content: result.clone(),
            })
            .await?;
            parts.push(part_tool_result(&call.id, status, &result));

            // Represent the approved call + its result in history so the model
            // can continue.
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: vec![call.clone()],
                tool_call_id: None,
            });
            history.push(ChatMessage::tool_result(&call.id, &result.to_string()));

            run_loop(
                store,
                exec,
                provider,
                sink,
                params,
                &msg,
                &mut history,
                &mut parts,
                &mut total_usage,
                &specs,
            )
            .await
        } else {
            let denied = json!({ "denied": true });
            store
                .complete_tool_call(&tref, "denied", &denied, 0)
                .await?;
            parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
            parts.push(part_tool_result(&call.id, "denied", &denied));
            sink.emit(AguiEvent::ToolCallResult {
                message_id: msg.id.clone(),
                tool_call_id: call.id.clone(),
                content: denied,
            })
            .await?;
            Ok(LoopResult::Completed)
        }
    }
    .await;

    finalize_run(store, sink, params, &msg, &parts, &total_usage, result).await
}

/// Shared completion / error handling for both [`run_turn`] and [`resume`].
async fn finalize_run<S, K>(
    store: &S,
    sink: &K,
    params: &RunParams,
    msg: &MessageRef,
    parts: &[Value],
    total_usage: &Usage,
    result: Result<LoopResult>,
) -> Result<()>
where
    S: ConversationStore,
    K: EventSink,
{
    match result {
        // The interrupt path already finalized the message + status.
        Ok(LoopResult::Interrupted) => Ok(()),
        Ok(LoopResult::Completed) => {
            sink.emit(AguiEvent::TextMessageEnd {
                message_id: msg.id.clone(),
            })
            .await?;
            store
                .finalize_assistant_message(
                    msg,
                    Value::Array(parts.to_vec()),
                    total_usage,
                    "complete",
                )
                .await?;
            store.set_conversation_status("idle").await?;
            sink.emit(AguiEvent::RunFinished {
                thread_id: params.thread_id.clone(),
                run_id: params.run_id.clone(),
                outcome: RunOutcome::Success,
                interrupt: None,
            })
            .await?;
            Ok(())
        }
        Err(e) => {
            // Best-effort teardown; ignore secondary errors.
            let _ = sink
                .emit(AguiEvent::RunError {
                    message: e.to_string(),
                    code: None,
                })
                .await;
            let _ = store.set_conversation_status("errored").await;
            let _ = store
                .finalize_assistant_message(
                    msg,
                    Value::Array(parts.to_vec()),
                    total_usage,
                    "errored",
                )
                .await;
            Err(e)
        }
    }
}

/// The core streaming loop: stream a turn, handle tools, repeat up to
/// `max_tool_turns`. Does not emit `TEXT_MESSAGE_END` or finalize — the caller
/// ([`finalize_run`]) does, so `run_turn` and `resume` share that logic.
#[allow(clippy::too_many_arguments)]
async fn run_loop<S, E, P, K>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    msg: &MessageRef,
    history: &mut Vec<ChatMessage>,
    parts: &mut Vec<Value>,
    total_usage: &mut Usage,
    specs: &[ToolSpec],
) -> Result<LoopResult>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
{
    for _turn in 0..params.max_tool_turns.max(1) {
        let outcome = stream_one_turn(provider, sink, &params.system, &msg.id, history, specs).await?;

        match outcome {
            TurnOutcome::Final { text, usage } => {
                total_usage.add(&usage);
                if !text.is_empty() {
                    parts.push(part_text(&text));
                }
                return Ok(LoopResult::Completed);
            }
            TurnOutcome::Tools {
                calls,
                usage,
                partial_text,
            } => {
                total_usage.add(&usage);
                if !partial_text.is_empty() {
                    parts.push(part_text(&partial_text));
                }
                // Represent the assistant's tool-call turn in history for the
                // next provider round.
                history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: partial_text.clone(),
                    tool_calls: calls.clone(),
                    tool_call_id: None,
                });

                for call in &calls {
                    let kind = specs
                        .iter()
                        .find(|s| s.name == call.name)
                        .map_or(ToolKind::Read, |s| s.kind);

                    if kind == ToolKind::Write && !params.auto_approve {
                        // Human-approval gate: record pending, interrupt, and
                        // finalize the message as still-streaming.
                        store.record_tool_call(msg, call, "pending").await?;
                        parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
                        sink.emit(AguiEvent::RunFinished {
                            thread_id: params.thread_id.clone(),
                            run_id: params.run_id.clone(),
                            outcome: RunOutcome::Interrupt,
                            interrupt: Some(Interrupt {
                                id: call.id.clone(),
                                reason: "human_approval".to_string(),
                                payload: json!({
                                    "toolCallId": call.id,
                                    "name": call.name,
                                    "arguments": call.arguments,
                                }),
                            }),
                        })
                        .await?;
                        store
                            .finalize_assistant_message(
                                msg,
                                Value::Array(parts.clone()),
                                total_usage,
                                "streaming",
                            )
                            .await?;
                        store.set_conversation_status("responding").await?;
                        return Ok(LoopResult::Interrupted);
                    }

                    // Read tool, or write tool with auto-approve: execute now.
                    let tref = store.record_tool_call(msg, call, "pending").await?;
                    sink.emit(AguiEvent::ToolCallStart {
                        tool_call_id: call.id.clone(),
                        tool_call_name: call.name.clone(),
                        parent_message_id: msg.id.clone(),
                    })
                    .await?;
                    sink.emit(AguiEvent::ToolCallArgs {
                        tool_call_id: call.id.clone(),
                        delta: call.arguments.to_string(),
                    })
                    .await?;
                    sink.emit(AguiEvent::ToolCallEnd {
                        tool_call_id: call.id.clone(),
                    })
                    .await?;

                    let (status, result) = execute_and_record(exec, store, &tref, call).await?;
                    sink.emit(AguiEvent::ToolCallResult {
                        message_id: msg.id.clone(),
                        tool_call_id: call.id.clone(),
                        content: result.clone(),
                    })
                    .await?;
                    parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
                    parts.push(part_tool_result(&call.id, status, &result));
                    history.push(ChatMessage::tool_result(&call.id, &result.to_string()));
                }
                // Loop again to let the model react to the tool results.
            }
        }
    }
    Ok(LoopResult::Completed)
}

/// Execute a tool, time it, and record completion. Returns the `(status,
/// result)` pair used for events and message parts.
async fn execute_and_record<E, S>(
    exec: &E,
    store: &S,
    tref: &ToolRef,
    call: &ToolCallReq,
) -> Result<(&'static str, Value)>
where
    E: ToolExecutor,
    S: ConversationStore,
{
    let start = Instant::now();
    let res = exec.execute(&call.name, call.arguments.clone()).await;
    let ms = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
    let (status, result) = match res {
        Ok(v) => ("success", v),
        Err(e) => ("error", json!({ "error": e.to_string() })),
    };
    store.complete_tool_call(tref, status, &result, ms).await?;
    Ok((status, result))
}

/// Stream a single provider turn, forwarding text deltas to the sink as
/// `TEXT_MESSAGE_CONTENT`, and return the assembled outcome.
///
/// Tool-call UI events are intentionally *not* forwarded from the raw delta
/// stream — they are emitted once from the caller after assembly, so the
/// human-approval gate can decide whether a tool call is streamed to the client
/// or converted into an interrupt.
async fn stream_one_turn<P, K>(
    provider: &P,
    sink: &K,
    system: &str,
    msg_id: &str,
    history: &[ChatMessage],
    specs: &[ToolSpec],
) -> Result<TurnOutcome>
where
    P: Provider,
    K: EventSink,
{
    use crate::agui::provider::AgentDelta;

    let (dtx, mut drx) = tokio::sync::mpsc::channel::<AgentDelta>(64);

    let provider_fut = async move {
        let r = provider.stream_turn(system, history, specs, &dtx).await;
        // Drop the sender so the forwarder's recv loop terminates.
        drop(dtx);
        r
    };

    let forward_fut = async move {
        while let Some(delta) = drx.recv().await {
            if let AgentDelta::TextDelta(text) = delta {
                sink.emit(AguiEvent::TextMessageContent {
                    message_id: msg_id.to_string(),
                    delta: text,
                })
                .await?;
            }
        }
        Ok::<(), Error>(())
    };

    let (out_res, fwd_res) = tokio::join!(provider_fut, forward_fut);
    // Surface a sink error (client disconnect) as the abort signal first.
    fwd_res?;
    out_res
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use crate::agui::protocol::{ResumePayload, ResumeItem};
    use crate::agui::provider::StubProvider;
    use std::sync::{Arc, Mutex};

    // ----- fakes -------------------------------------------------------------

    #[derive(Default)]
    struct SinkState {
        events: Vec<AguiEvent>,
    }

    #[derive(Clone, Default)]
    struct VecSink(Arc<Mutex<SinkState>>);

    impl VecSink {
        fn names(&self) -> Vec<String> {
            self.0
                .lock()
                .unwrap()
                .events
                .iter()
                .map(|e| e.event_name().to_string())
                .collect()
        }
        fn events(&self) -> Vec<AguiEvent> {
            self.0.lock().unwrap().events.clone()
        }
    }

    #[async_trait::async_trait]
    impl EventSink for VecSink {
        async fn emit(&self, ev: AguiEvent) -> Result<()> {
            self.0.lock().unwrap().events.push(ev);
            Ok(())
        }
    }

    struct ToolRecord {
        tool_call_id: String,
        name: String,
        arguments: Value,
        message_id: String,
        status: String,
    }

    #[derive(Default)]
    struct StoreState {
        history: Vec<ChatMessage>,
        msg_counter: usize,
        tools: Vec<ToolRecord>,
        status: String,
        finalized: Vec<(String, String)>, // (message_id, status)
    }

    #[derive(Clone, Default)]
    struct FakeStore(Arc<Mutex<StoreState>>);

    impl FakeStore {
        fn with_user(msg: &str) -> Self {
            let s = Self::default();
            s.0.lock().unwrap().history.push(ChatMessage::text("user", msg));
            s
        }
        fn status(&self) -> String {
            self.0.lock().unwrap().status.clone()
        }
        fn tool_status(&self, id: &str) -> Option<String> {
            self.0
                .lock()
                .unwrap()
                .tools
                .iter()
                .find(|t| t.tool_call_id == id)
                .map(|t| t.status.clone())
        }
    }

    #[async_trait::async_trait]
    impl ConversationStore for FakeStore {
        async fn load_history(&self) -> Result<Vec<ChatMessage>> {
            Ok(self.0.lock().unwrap().history.clone())
        }
        async fn append_user_message(&self, text: &str) -> Result<MessageRef> {
            let mut s = self.0.lock().unwrap();
            s.history.push(ChatMessage::text("user", text));
            s.msg_counter += 1;
            Ok(MessageRef {
                id: format!("umsg_{}", s.msg_counter),
            })
        }
        async fn begin_assistant_message(&self, _p: &str, _m: &str) -> Result<MessageRef> {
            let mut s = self.0.lock().unwrap();
            s.msg_counter += 1;
            Ok(MessageRef {
                id: format!("msg_{}", s.msg_counter),
            })
        }
        async fn record_tool_call(
            &self,
            msg: &MessageRef,
            call: &ToolCallReq,
            status: &str,
        ) -> Result<ToolRef> {
            let mut s = self.0.lock().unwrap();
            s.tools.push(ToolRecord {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                message_id: msg.id.clone(),
                status: status.to_string(),
            });
            Ok(ToolRef { id: call.id.clone() })
        }
        async fn complete_tool_call(
            &self,
            tool: &ToolRef,
            status: &str,
            _result: &Value,
            _ms: i64,
        ) -> Result<()> {
            let mut s = self.0.lock().unwrap();
            if let Some(t) = s.tools.iter_mut().find(|t| t.tool_call_id == tool.id) {
                t.status = status.to_string();
            }
            Ok(())
        }
        async fn finalize_assistant_message(
            &self,
            msg: &MessageRef,
            _parts: Value,
            _usage: &Usage,
            status: &str,
        ) -> Result<()> {
            self.0
                .lock()
                .unwrap()
                .finalized
                .push((msg.id.clone(), status.to_string()));
            Ok(())
        }
        async fn find_pending_tool_call(
            &self,
            tool_call_id: &str,
        ) -> Result<Option<PendingToolCall>> {
            let s = self.0.lock().unwrap();
            Ok(s.tools
                .iter()
                .find(|t| t.tool_call_id == tool_call_id && t.status == "pending")
                .map(|t| PendingToolCall {
                    tool_call_id: t.tool_call_id.clone(),
                    name: t.name.clone(),
                    arguments: t.arguments.clone(),
                    message_id: t.message_id.clone(),
                }))
        }
        async fn set_conversation_status(&self, status: &str) -> Result<()> {
            self.0.lock().unwrap().status = status.to_string();
            Ok(())
        }
    }

    struct FakeExec;

    #[async_trait::async_trait]
    impl ToolExecutor for FakeExec {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![
                ToolSpec {
                    name: "lookup".to_string(),
                    description: "read data".to_string(),
                    parameters: json!({"type": "object"}),
                    kind: ToolKind::Read,
                },
                ToolSpec {
                    name: "save_note".to_string(),
                    description: "write data".to_string(),
                    parameters: json!({"type": "object"}),
                    kind: ToolKind::Write,
                },
            ]
        }
        async fn execute(&self, name: &str, _args: Value) -> Result<Value> {
            Ok(json!({ "ok": name }))
        }
    }

    fn params(auto_approve: bool) -> RunParams {
        RunParams {
            system: "you are a test agent".to_string(),
            run_id: "run1".to_string(),
            thread_id: "thread1".to_string(),
            auto_approve,
            max_tool_turns: 5,
        }
    }

    // ----- tests -------------------------------------------------------------

    #[tokio::test]
    async fn happy_text_path_event_order() {
        let store = FakeStore::with_user("just chatting please");
        let sink = VecSink::default();
        let provider = StubProvider::with_reply("hi there friend");

        run_turn(&store, &FakeExec, &provider, &sink, &params(false))
            .await
            .unwrap();

        let names = sink.names();
        assert_eq!(names.first().unwrap(), "RUN_STARTED");
        assert_eq!(names[1], "TEXT_MESSAGE_START");
        assert!(names.contains(&"TEXT_MESSAGE_CONTENT".to_string()));
        assert!(!names.contains(&"TOOL_CALL_START".to_string()));
        assert_eq!(names[names.len() - 2], "TEXT_MESSAGE_END");
        assert_eq!(names.last().unwrap(), "RUN_FINISHED");
        assert_eq!(store.status(), "idle");
    }

    #[tokio::test]
    async fn auto_approved_tool_path() {
        let store = FakeStore::with_user("please update the note");
        let sink = VecSink::default();
        let provider = StubProvider::new();

        run_turn(&store, &FakeExec, &provider, &sink, &params(true))
            .await
            .unwrap();

        let names = sink.names();
        assert_eq!(names[0], "RUN_STARTED");
        assert!(names.contains(&"TOOL_CALL_START".to_string()));
        assert!(names.contains(&"TOOL_CALL_ARGS".to_string()));
        assert!(names.contains(&"TOOL_CALL_END".to_string()));
        assert!(names.contains(&"TOOL_CALL_RESULT".to_string()));
        assert_eq!(names.last().unwrap(), "RUN_FINISHED");
        // Success (not interrupt).
        match sink.events().last().unwrap() {
            AguiEvent::RunFinished { outcome, .. } => {
                assert!(matches!(outcome, RunOutcome::Success));
            }
            _ => panic!("expected RunFinished"),
        }
        assert_eq!(store.status(), "idle");
    }

    #[tokio::test]
    async fn write_tool_interrupts_and_records_pending() {
        let store = FakeStore::with_user("please write a note");
        let sink = VecSink::default();
        let provider = StubProvider::new();

        run_turn(&store, &FakeExec, &provider, &sink, &params(false))
            .await
            .unwrap();

        // Ends in an interrupt.
        match sink.events().last().unwrap() {
            AguiEvent::RunFinished {
                outcome, interrupt, ..
            } => {
                assert!(matches!(outcome, RunOutcome::Interrupt));
                assert_eq!(interrupt.as_ref().unwrap().reason, "human_approval");
            }
            _ => panic!("expected interrupt RunFinished"),
        }
        assert_eq!(store.status(), "responding");
        assert_eq!(
            store.tool_status("call_stub_save_note").as_deref(),
            Some("pending")
        );
    }

    #[tokio::test]
    async fn resume_approved_continues_to_success() {
        let store = FakeStore::with_user("please write a note");
        let sink = VecSink::default();
        let provider = StubProvider::new();

        run_turn(&store, &FakeExec, &provider, &sink, &params(false))
            .await
            .unwrap();
        assert_eq!(store.status(), "responding");

        let resume_sink = VecSink::default();
        let item = ResumeItem {
            interrupt_id: "call_stub_save_note".to_string(),
            payload: ResumePayload { approved: true },
        };
        resume(&store, &FakeExec, &provider, &resume_sink, &params(false), &item)
            .await
            .unwrap();

        let names = resume_sink.names();
        assert!(names.contains(&"TOOL_CALL_RESULT".to_string()));
        assert_eq!(names.last().unwrap(), "RUN_FINISHED");
        match resume_sink.events().last().unwrap() {
            AguiEvent::RunFinished { outcome, .. } => {
                assert!(matches!(outcome, RunOutcome::Success));
            }
            _ => panic!("expected RunFinished"),
        }
        assert_eq!(store.status(), "idle");
        assert_eq!(
            store.tool_status("call_stub_save_note").as_deref(),
            Some("success")
        );
    }

    #[tokio::test]
    async fn resume_denied_ends_denied() {
        let store = FakeStore::with_user("please write a note");
        let sink = VecSink::default();
        let provider = StubProvider::new();

        run_turn(&store, &FakeExec, &provider, &sink, &params(false))
            .await
            .unwrap();

        let resume_sink = VecSink::default();
        let item = ResumeItem {
            interrupt_id: "call_stub_save_note".to_string(),
            payload: ResumePayload { approved: false },
        };
        resume(&store, &FakeExec, &provider, &resume_sink, &params(false), &item)
            .await
            .unwrap();

        let names = resume_sink.names();
        assert!(names.contains(&"TOOL_CALL_RESULT".to_string()));
        assert_eq!(names.last().unwrap(), "RUN_FINISHED");
        assert_eq!(store.status(), "idle");
        assert_eq!(
            store.tool_status("call_stub_save_note").as_deref(),
            Some("denied")
        );
    }
}
