//! # Reusable agent run-loop
//!
//! [`run_turn`] and [`resume`] drive a streaming agent turn end to end, wiring a
//! [`Provider`] to an app-supplied [`ConversationStore`] (persistence) and
//! [`ToolExecutor`] (tool implementations), emitting [`AguiEvent`]s into an
//! [`EventSink`] as it goes.
//!
//! The loop is deliberately free of any application concepts (no citation
//! parsing, personas, scopes, or concrete tables). Everything app-specific
//! arrives through the traits below; the app does its own post-processing
//! inside its [`ConversationStore::finalize_assistant_message`] implementation.
//!
//! Before each tool call runs, the loop consults an app-supplied
//! [`ToolAuthorizer`], which may [`Allow`](ToolDecision::Allow),
//! [`Deny`](ToolDecision::Deny) (a model-visible refusal), or
//! [`RequireApproval`](ToolDecision::RequireApproval) (force the human-approval
//! interrupt). Pass [`AllowAll`] to keep the built-in write-approval behavior.
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
use tracing::{debug, error, info, instrument};

use crate::agui::protocol::{
    part_text, part_tool_result, part_tool_use, AguiEvent, Interrupt, ResumeItem, RunOutcome,
};
use crate::agui::provider::{
    ChatMessage, Provider, ToolCallReq, ToolKind, ToolSpec, TurnOutcome, Usage,
};
use crate::agui::subagent::{SubagentCtx, SubagentRegistry, SubagentStep};
use crate::agui::transport::EventSink;
use crate::{Error, Result};

/// Key under which a bubbled-up subagent's suspended state is stashed in the
/// parent pending delegation call's arguments (so `resume` can route back).
const SUBAGENT_STATE_KEY: &str = "__subagent_state";

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

/// The outcome of an authorization check for a single tool call.
///
/// Returned by [`ToolAuthorizer::authorize`], consulted by the run-loop *before*
/// every tool call — ahead of the built-in [`ToolKind::Write`] +
/// [`RunParams::auto_approve`] approval gate.
#[derive(Debug, Clone)]
pub enum ToolDecision {
    /// Proceed. The call is still subject to the normal write/auto-approve
    /// approval gate afterwards.
    Allow,
    /// Refuse without executing. `reason` is surfaced to the model as the tool
    /// result (`{"denied": true, "reason": ...}`) so it can react and continue.
    Deny { reason: String },
    /// Route into the human-approval interrupt path regardless of the tool's
    /// [`ToolKind`]. `reason` is echoed as the interrupt reason.
    RequireApproval { reason: String },
}

/// App-supplied per-call authorization for tool calls.
///
/// The run-loop calls [`authorize`](ToolAuthorizer::authorize) for every tool
/// the model requests before it runs. This is the seam for principal/scope
/// checks: the implementor captures whatever request-scoped context it needs
/// (authenticated user, scopes, tenant) at construction — the same injection
/// pattern used by the app's [`ConversationStore`] — since the generic run-loop
/// has no notion of a user.
///
/// Pass [`AllowAll`] to opt out (reproduces the pre-authorization behavior).
#[async_trait::async_trait]
pub trait ToolAuthorizer: Send + Sync {
    /// Decide whether `call` (of the given `kind`) may run.
    ///
    /// # Errors
    /// An `Err` aborts the run (surfaced as `RUN_ERROR`); use [`ToolDecision::Deny`]
    /// for an ordinary, model-visible refusal.
    async fn authorize(&self, call: &ToolCallReq, kind: ToolKind) -> Result<ToolDecision>;
}

/// A [`ToolAuthorizer`] that permits every call, preserving the behavior from
/// before authorization existed (the built-in write/approval gate still applies).
pub struct AllowAll;

#[async_trait::async_trait]
impl ToolAuthorizer for AllowAll {
    async fn authorize(&self, _call: &ToolCallReq, _kind: ToolKind) -> Result<ToolDecision> {
        Ok(ToolDecision::Allow)
    }
}

/// Forwarding impl so `Box<dyn ToolExecutor>` is a `Sized` [`ToolExecutor`] —
/// needed to store heterogeneous executors (e.g. in a composite or a subagent
/// registry) and still satisfy `run_turn`'s `Sized` generic bounds.
#[async_trait::async_trait]
impl<T: ?Sized + ToolExecutor> ToolExecutor for Box<T> {
    fn specs(&self) -> Vec<ToolSpec> {
        (**self).specs()
    }
    async fn execute(&self, name: &str, args: Value) -> Result<Value> {
        (**self).execute(name, args).await
    }
}

/// Forwarding impl so `Box<dyn ToolAuthorizer>` is a `Sized` [`ToolAuthorizer`].
#[async_trait::async_trait]
impl<T: ?Sized + ToolAuthorizer> ToolAuthorizer for Box<T> {
    async fn authorize(&self, call: &ToolCallReq, kind: ToolKind) -> Result<ToolDecision> {
        (**self).authorize(call, kind).await
    }
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
#[instrument(
    target = "loco_rs::agui",
    name = "agui.run_turn",
    skip_all,
    fields(run_id = %params.run_id, thread_id = %params.thread_id, model = %provider.model_id()),
)]
pub async fn run_turn<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
    A: ToolAuthorizer,
{
    run_turn_impl(store, exec, provider, sink, params, authz, None).await
}

/// Like [`run_turn`], but delegation tool calls that match a subagent in
/// `subagents` are run as nested agents; a subagent's human-approval need
/// bubbles up as a parent interrupt (see [`crate::agui::subagent`]).
///
/// # Errors
/// As [`run_turn`].
#[instrument(
    target = "loco_rs::agui",
    name = "agui.run_turn_subagents",
    skip_all,
    fields(run_id = %params.run_id, thread_id = %params.thread_id, model = %provider.model_id()),
)]
pub async fn run_turn_with_subagents<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
    subagents: &SubagentRegistry,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
    A: ToolAuthorizer,
{
    run_turn_impl(store, exec, provider, sink, params, authz, Some(subagents)).await
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_impl<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
    subagents: Option<&SubagentRegistry>,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
    A: ToolAuthorizer,
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
    let specs = merged_specs(exec, subagents);
    let mut parts: Vec<Value> = Vec::new();
    let mut total_usage = Usage::default();

    let result = run_loop(
        store,
        exec,
        provider,
        sink,
        params,
        authz,
        subagents,
        &msg,
        &mut history,
        &mut parts,
        &mut total_usage,
        &specs,
    )
    .await;

    finalize_run(store, sink, params, &msg, &parts, &total_usage, result).await
}

/// The tool specs the provider should see: the executor's tools plus one entry
/// per subagent (so the model can call them). Subagent specs come from the
/// registry; a name collision keeps the executor's spec (first-wins).
fn merged_specs<E: ToolExecutor>(exec: &E, subagents: Option<&SubagentRegistry>) -> Vec<ToolSpec> {
    let mut specs = exec.specs();
    if let Some(reg) = subagents {
        let have: std::collections::BTreeSet<String> =
            specs.iter().map(|s| s.name.clone()).collect();
        for s in reg.specs() {
            if !have.contains(&s.name) {
                specs.push(s);
            }
        }
    }
    specs
}

/// Resume a previously interrupted run by answering its approval gate.
///
/// # Errors
/// Errors if no `pending` tool call matches `item.interrupt_id`, or on
/// provider/store/sink failure (same error handling as [`run_turn`]).
#[instrument(
    target = "loco_rs::agui",
    name = "agui.resume",
    skip_all,
    fields(run_id = %params.run_id, thread_id = %params.thread_id, interrupt_id = %item.interrupt_id),
)]
pub async fn resume<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
    item: &ResumeItem,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
    A: ToolAuthorizer,
{
    resume_impl(store, exec, provider, sink, params, authz, None, item).await
}

/// Like [`resume`], but if the pending call is a subagent delegation the
/// approval is routed back into the suspended subagent (approval bubble-up).
///
/// # Errors
/// As [`resume`].
#[instrument(
    target = "loco_rs::agui",
    name = "agui.resume_subagents",
    skip_all,
    fields(run_id = %params.run_id, thread_id = %params.thread_id, interrupt_id = %item.interrupt_id),
)]
#[allow(clippy::too_many_arguments)]
pub async fn resume_with_subagents<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
    subagents: &SubagentRegistry,
    item: &ResumeItem,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
    A: ToolAuthorizer,
{
    resume_impl(store, exec, provider, sink, params, authz, Some(subagents), item).await
}

#[allow(clippy::too_many_arguments)]
async fn resume_impl<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
    subagents: Option<&SubagentRegistry>,
    item: &ResumeItem,
) -> Result<()>
where
    S: ConversationStore,
    E: ToolExecutor,
    P: Provider,
    K: EventSink,
    A: ToolAuthorizer,
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
    let specs = merged_specs(exec, subagents);
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

    // Subagent-delegation resume: route the approval back into the suspended
    // child, then continue (or bubble again).
    if let Some(agent) = pending
        .arguments
        .get(SUBAGENT_STATE_KEY)
        .and(subagents)
        .and_then(|r| r.get(&pending.name).cloned())
    {
        let reg = subagents.expect("registry present when subagent resolved");
        let state = pending.arguments[SUBAGENT_STATE_KEY].clone();
        let display_call = ToolCallReq {
            id: pending.tool_call_id.clone(),
            name: pending.name.clone(),
            arguments: json!({ "input": pending.arguments.get("input").cloned().unwrap_or(Value::Null) }),
        };
        let ctx = SubagentCtx {
            depth: 1,
            max_depth: reg.max_depth(),
        };
        let child_sink = reg.child_sink();
        let result: Result<LoopResult> = async {
            match agent
                .resume_step(state, item.payload.approved, &ctx, child_sink.as_ref())
                .await
            {
                Ok(SubagentStep::Interrupted { interrupt, state }) => {
                    bubble_subagent_interrupt(
                        store, sink, params, &msg, &display_call, interrupt, state, &mut parts,
                        &total_usage,
                    )
                    .await?;
                    Ok(LoopResult::Interrupted)
                }
                other => {
                    let (status, res) = match other {
                        Ok(SubagentStep::Done(out)) => ("success", json!({ "output": out.text })),
                        Err(e) => ("error", json!({ "error": e.to_string() })),
                        Ok(SubagentStep::Interrupted { .. }) => unreachable!(),
                    };
                    store.complete_tool_call(&tref, status, &res, 0).await?;
                    sink.emit(AguiEvent::ToolCallResult {
                        message_id: msg.id.clone(),
                        tool_call_id: display_call.id.clone(),
                        content: res.clone(),
                    })
                    .await?;
                    parts.push(part_tool_use(
                        &display_call.id,
                        &display_call.name,
                        &display_call.arguments,
                    ));
                    parts.push(part_tool_result(&display_call.id, status, &res));
                    history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: String::new(),
                        tool_calls: vec![display_call.clone()],
                        tool_call_id: None,
                    });
                    history.push(ChatMessage::tool_result(&display_call.id, &res.to_string()));
                    run_loop(
                        store, exec, provider, sink, params, authz, subagents, &msg, &mut history,
                        &mut parts, &mut total_usage, &specs,
                    )
                    .await
                }
            }
        }
        .await;
        return finalize_run(store, sink, params, &msg, &parts, &total_usage, result).await;
    }

    let result: Result<LoopResult> = async {
        if item.payload.approved {
            // Defense in depth: re-authorize the approved call. The principal or
            // scope may have changed between the interrupt and the approval, so a
            // human "yes" does not override a `Deny`.
            let kind = specs
                .iter()
                .find(|s| s.name == call.name)
                .map_or(ToolKind::Write, |s| s.kind);
            let (status, result) = match authz.authorize(&call, kind).await? {
                ToolDecision::Deny { reason } => {
                    let denied = json!({ "denied": true, "reason": reason });
                    store.complete_tool_call(&tref, "denied", &denied, 0).await?;
                    ("denied", denied)
                }
                // An approved call is already past its approval gate, so treat a
                // repeated `RequireApproval` as an allow rather than looping.
                ToolDecision::Allow | ToolDecision::RequireApproval { .. } => {
                    execute_and_record(exec, store, &tref, &call).await?
                }
            };
            parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
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
                authz,
                subagents,
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
            error!(target: "loco_rs::agui", error = %e, run_id = %params.run_id, "agent run failed");
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
async fn run_loop<S, E, P, K, A>(
    store: &S,
    exec: &E,
    provider: &P,
    sink: &K,
    params: &RunParams,
    authz: &A,
    subagents: Option<&SubagentRegistry>,
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
    A: ToolAuthorizer,
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

                    // Authorization gate — consulted before the built-in
                    // write/auto-approve gate. Decides deny / require-approval /
                    // allow for this specific call and principal.
                    let interrupt_reason = match authz.authorize(call, kind).await? {
                        ToolDecision::Deny { reason } => {
                            // Hard refusal: never execute. Record + surface a
                            // `denied` result the model can see, then move on.
                            debug!(
                                target: "loco_rs::agui",
                                tool = %call.name, %reason, "authz denied tool call"
                            );
                            let denied = json!({ "denied": true, "reason": reason });
                            let tref = store.record_tool_call(msg, call, "denied").await?;
                            store.complete_tool_call(&tref, "denied", &denied, 0).await?;
                            sink.emit(AguiEvent::ToolCallResult {
                                message_id: msg.id.clone(),
                                tool_call_id: call.id.clone(),
                                content: denied.clone(),
                            })
                            .await?;
                            parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
                            parts.push(part_tool_result(&call.id, "denied", &denied));
                            history.push(ChatMessage::tool_result(&call.id, &denied.to_string()));
                            continue;
                        }
                        // Force the human-approval path regardless of ToolKind.
                        ToolDecision::RequireApproval { reason } => Some(reason),
                        // Allowed: fall back to the built-in write/approval gate.
                        ToolDecision::Allow => {
                            if kind == ToolKind::Write && !params.auto_approve {
                                Some("human_approval".to_string())
                            } else {
                                None
                            }
                        }
                    };

                    if let Some(reason) = interrupt_reason {
                        // Human-approval gate: record pending, interrupt, and
                        // finalize the message as still-streaming.
                        info!(
                            target: "loco_rs::agui",
                            tool = %call.name, %reason, "raising approval interrupt"
                        );
                        store.record_tool_call(msg, call, "pending").await?;
                        parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
                        sink.emit(AguiEvent::RunFinished {
                            thread_id: params.thread_id.clone(),
                            run_id: params.run_id.clone(),
                            outcome: RunOutcome::Interrupt,
                            interrupt: Some(Interrupt {
                                id: call.id.clone(),
                                reason,
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

                    // Subagent delegation: run the matching child as a nested
                    // agent. A child's human-approval need bubbles up as a parent
                    // interrupt whose pending call carries the child's suspended
                    // state for `resume` to route back into (see `subagent`).
                    if let Some(agent) = subagents.and_then(|r| r.get(&call.name).cloned()) {
                        let reg = subagents.expect("registry present when agent resolved");
                        emit_tool_call_frames(sink, &msg.id, call).await?;
                        let input = call
                            .arguments
                            .get("input")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let ctx = SubagentCtx {
                            depth: 1,
                            max_depth: reg.max_depth(),
                        };
                        let child_sink = reg.child_sink();
                        match agent.start(input, &ctx, child_sink.as_ref()).await {
                            Ok(SubagentStep::Done(out)) => {
                                let result = json!({ "output": out.text });
                                record_delegation_result(
                                    store, sink, &msg.id, call, "success", &result, parts, history,
                                )
                                .await?;
                            }
                            Ok(SubagentStep::Interrupted { interrupt, state }) => {
                                bubble_subagent_interrupt(
                                    store, sink, params, msg, call, interrupt, state, parts,
                                    total_usage,
                                )
                                .await?;
                                return Ok(LoopResult::Interrupted);
                            }
                            Err(e) => {
                                let result = json!({ "error": e.to_string() });
                                record_delegation_result(
                                    store, sink, &msg.id, call, "error", &result, parts, history,
                                )
                                .await?;
                            }
                        }
                        continue;
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
    debug!(
        target: "loco_rs::agui",
        tool = %call.name, status, duration_ms = ms, "tool call complete"
    );
    store.complete_tool_call(tref, status, &result, ms).await?;
    Ok((status, result))
}

/// Emit the `TOOL_CALL_START/ARGS/END` frames for a tool (or subagent) call.
async fn emit_tool_call_frames<K: EventSink>(
    sink: &K,
    msg_id: &str,
    call: &ToolCallReq,
) -> Result<()> {
    sink.emit(AguiEvent::ToolCallStart {
        tool_call_id: call.id.clone(),
        tool_call_name: call.name.clone(),
        parent_message_id: msg_id.to_string(),
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
    .await
}

/// Record + surface a completed subagent delegation's result (success or error)
/// and thread it into `parts`/`history` so the parent model reacts to it.
#[allow(clippy::too_many_arguments)]
async fn record_delegation_result<S, K>(
    store: &S,
    sink: &K,
    msg_id: &str,
    call: &ToolCallReq,
    status: &'static str,
    result: &Value,
    parts: &mut Vec<Value>,
    history: &mut Vec<ChatMessage>,
) -> Result<()>
where
    S: ConversationStore,
    K: EventSink,
{
    let tref = store.record_tool_call(&MessageRef { id: msg_id.to_string() }, call, "pending").await?;
    store.complete_tool_call(&tref, status, result, 0).await?;
    sink.emit(AguiEvent::ToolCallResult {
        message_id: msg_id.to_string(),
        tool_call_id: call.id.clone(),
        content: result.clone(),
    })
    .await?;
    parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
    parts.push(part_tool_result(&call.id, status, result));
    history.push(ChatMessage::tool_result(&call.id, &result.to_string()));
    Ok(())
}

/// Bubble a subagent's approval interrupt up to the parent: persist a `pending`
/// delegation call carrying the child's suspended `state`, emit the parent
/// `RUN_FINISHED(Interrupt)` (keyed by the parent delegation id), and finalize
/// the message as still-streaming.
#[allow(clippy::too_many_arguments)]
async fn bubble_subagent_interrupt<S, K>(
    store: &S,
    sink: &K,
    params: &RunParams,
    msg: &MessageRef,
    call: &ToolCallReq,
    interrupt: Interrupt,
    state: Value,
    parts: &mut Vec<Value>,
    total_usage: &Usage,
) -> Result<()>
where
    S: ConversationStore,
    K: EventSink,
{
    // The pending call is keyed by the parent delegation id and carries the
    // child's suspended state so `resume` can re-enter the subagent.
    let pending_call = ToolCallReq {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: json!({
            "input": call.arguments.get("input").cloned().unwrap_or(Value::Null),
            SUBAGENT_STATE_KEY: state,
        }),
    };
    store.record_tool_call(msg, &pending_call, "pending").await?;
    parts.push(part_tool_use(&call.id, &call.name, &call.arguments));
    sink.emit(AguiEvent::RunFinished {
        thread_id: params.thread_id.clone(),
        run_id: params.run_id.clone(),
        outcome: RunOutcome::Interrupt,
        interrupt: Some(Interrupt {
            id: call.id.clone(), // parent delegation id — the resume key
            reason: interrupt.reason,
            payload: interrupt.payload,
        }),
    })
    .await?;
    store
        .finalize_assistant_message(msg, Value::Array(parts.clone()), total_usage, "streaming")
        .await?;
    store.set_conversation_status("responding").await?;
    info!(target: "loco_rs::agui", subagent = %call.name, "subagent approval interrupt bubbled up");
    Ok(())
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
    use crate::agui::provider::{AgentDelta, StubProvider};
    use crate::agui::subagent::LocalSubagent;
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

    /// Denies every call with a fixed reason.
    struct DenyAll;
    #[async_trait::async_trait]
    impl ToolAuthorizer for DenyAll {
        async fn authorize(&self, _call: &ToolCallReq, _kind: ToolKind) -> Result<ToolDecision> {
            Ok(ToolDecision::Deny {
                reason: "not permitted".to_string(),
            })
        }
    }

    /// Forces the approval path for every call, regardless of ToolKind.
    struct RequireApprovalAll;
    #[async_trait::async_trait]
    impl ToolAuthorizer for RequireApprovalAll {
        async fn authorize(&self, _call: &ToolCallReq, _kind: ToolKind) -> Result<ToolDecision> {
            Ok(ToolDecision::RequireApproval {
                reason: "needs_review".to_string(),
            })
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

        run_turn(&store, &FakeExec, &provider, &sink, &params(false), &AllowAll)
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

        run_turn(&store, &FakeExec, &provider, &sink, &params(true), &AllowAll)
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

        run_turn(&store, &FakeExec, &provider, &sink, &params(false), &AllowAll)
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

        run_turn(&store, &FakeExec, &provider, &sink, &params(false), &AllowAll)
            .await
            .unwrap();
        assert_eq!(store.status(), "responding");

        let resume_sink = VecSink::default();
        let item = ResumeItem {
            interrupt_id: "call_stub_save_note".to_string(),
            payload: ResumePayload { approved: true },
        };
        resume(&store, &FakeExec, &provider, &resume_sink, &params(false), &AllowAll, &item)
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

        run_turn(&store, &FakeExec, &provider, &sink, &params(false), &AllowAll)
            .await
            .unwrap();

        let resume_sink = VecSink::default();
        let item = ResumeItem {
            interrupt_id: "call_stub_save_note".to_string(),
            payload: ResumePayload { approved: false },
        };
        resume(&store, &FakeExec, &provider, &resume_sink, &params(false), &AllowAll, &item)
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

    #[tokio::test]
    async fn authz_deny_refuses_without_executing() {
        let store = FakeStore::with_user("please write a note");
        let sink = VecSink::default();
        let provider = StubProvider::new();

        // auto_approve = true so the *only* thing that can stop execution is the
        // authorizer, not the built-in write gate.
        run_turn(&store, &FakeExec, &provider, &sink, &params(true), &DenyAll)
            .await
            .unwrap();

        let names = sink.names();
        // Never executed: no TOOL_CALL_START/ARGS/END were emitted.
        assert!(!names.contains(&"TOOL_CALL_START".to_string()));
        // But the refusal is surfaced to the model as a tool result.
        assert!(names.contains(&"TOOL_CALL_RESULT".to_string()));
        // The run still completes successfully (deny is not an interrupt).
        match sink.events().last().unwrap() {
            AguiEvent::RunFinished { outcome, .. } => {
                assert!(matches!(outcome, RunOutcome::Success));
            }
            _ => panic!("expected RunFinished"),
        }
        assert_eq!(store.status(), "idle");
        assert_eq!(
            store.tool_status("call_stub_save_note").as_deref(),
            Some("denied")
        );
        // The emitted result carries the denial marker + reason.
        let denied_content = sink.events().into_iter().find_map(|e| match e {
            AguiEvent::ToolCallResult { content, .. } => Some(content),
            _ => None,
        });
        let content = denied_content.expect("a TOOL_CALL_RESULT");
        assert_eq!(content["denied"], json!(true));
        assert_eq!(content["reason"], json!("not permitted"));
    }

    #[tokio::test]
    async fn authz_require_approval_overrides_auto_approve() {
        let store = FakeStore::with_user("please write a note");
        let sink = VecSink::default();
        let provider = StubProvider::new();

        // With auto_approve = true the built-in gate would execute the write tool
        // outright; RequireApproval must still force the interrupt.
        run_turn(
            &store,
            &FakeExec,
            &provider,
            &sink,
            &params(true),
            &RequireApprovalAll,
        )
        .await
        .unwrap();

        match sink.events().last().unwrap() {
            AguiEvent::RunFinished {
                outcome, interrupt, ..
            } => {
                assert!(matches!(outcome, RunOutcome::Interrupt));
                assert_eq!(interrupt.as_ref().unwrap().reason, "needs_review");
            }
            _ => panic!("expected interrupt RunFinished"),
        }
        assert_eq!(store.status(), "responding");
        assert_eq!(
            store.tool_status("call_stub_save_note").as_deref(),
            Some("pending")
        );
    }

    // ----- Stage 3: subagent approval bubble-up -----------------------------

    /// A parent provider that delegates to `tool` on the first (user) turn and
    /// returns a final answer once a tool result is in history.
    struct DelegatingProvider {
        tool: String,
    }
    #[async_trait::async_trait]
    impl Provider for DelegatingProvider {
        fn model_id(&self) -> String {
            "parent-model".to_string()
        }
        async fn run_turn(&self, _s: &str, _h: &[ChatMessage], _t: &[ToolSpec]) -> Result<TurnOutcome> {
            unreachable!("streaming only in this test")
        }
        async fn stream_turn(
            &self,
            _system: &str,
            history: &[ChatMessage],
            _tools: &[ToolSpec],
            tx: &tokio::sync::mpsc::Sender<AgentDelta>,
        ) -> Result<TurnOutcome> {
            let last_is_user = history.last().map(|m| m.role == "user").unwrap_or(false);
            if last_is_user {
                Ok(TurnOutcome::Tools {
                    calls: vec![ToolCallReq {
                        id: "del_1".to_string(),
                        name: self.tool.clone(),
                        arguments: json!({ "input": "please update the record" }),
                    }],
                    usage: Usage::default(),
                    partial_text: String::new(),
                })
            } else {
                let _ = tx.send(AgentDelta::TextDelta("all done".to_string())).await;
                Ok(TurnOutcome::Final {
                    text: "all done".to_string(),
                    usage: Usage::default(),
                })
            }
        }
    }

    // The subagent's own local write tool.
    struct WriteExec;
    #[async_trait::async_trait]
    impl ToolExecutor for WriteExec {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![ToolSpec {
                name: "save".to_string(),
                description: "save".to_string(),
                parameters: json!({"type": "object"}),
                kind: ToolKind::Write,
            }]
        }
        async fn execute(&self, name: &str, _args: Value) -> Result<Value> {
            Ok(json!({ "saved": name }))
        }
    }

    fn worker_registry() -> SubagentRegistry {
        let mut reg = SubagentRegistry::default();
        reg.register(LocalSubagent {
            name: "worker".to_string(),
            description: "does work with a write tool".to_string(),
            system: "you are a worker".to_string(),
            provider: StubProvider::new(),
            exec: WriteExec,
            authz: AllowAll,
            max_tool_turns: 4,
        });
        reg
    }

    #[tokio::test]
    async fn subagent_write_bubbles_up_and_resumes_to_success() {
        let store = FakeStore::with_user("delegate please");
        let sink = VecSink::default();
        let provider = DelegatingProvider { tool: "worker".to_string() };
        let reg = worker_registry();

        // 1) Parent delegates → subagent's write tool interrupts → bubbles up.
        run_turn_with_subagents(
            &store,
            &FakeExec,
            &provider,
            &sink,
            &params(false),
            &AllowAll,
            &reg,
        )
        .await
        .unwrap();

        match sink.events().last().unwrap() {
            AguiEvent::RunFinished {
                outcome, interrupt, ..
            } => {
                assert!(matches!(outcome, RunOutcome::Interrupt));
                let itr = interrupt.as_ref().unwrap();
                assert_eq!(itr.reason, "subagent_approval");
                assert_eq!(itr.id, "del_1"); // keyed by the parent delegation id
            }
            _ => panic!("expected a bubbled subagent interrupt"),
        }
        assert_eq!(store.status(), "responding");
        assert_eq!(store.tool_status("del_1").as_deref(), Some("pending"));

        // 2) Approve → resume routes into the child, which completes; the parent
        // then continues to a successful finish.
        let resume_sink = VecSink::default();
        let item = ResumeItem {
            interrupt_id: "del_1".to_string(),
            payload: ResumePayload { approved: true },
        };
        resume_with_subagents(
            &store,
            &FakeExec,
            &provider,
            &resume_sink,
            &params(false),
            &AllowAll,
            &reg,
            &item,
        )
        .await
        .unwrap();

        let names = resume_sink.names();
        assert!(names.contains(&"TOOL_CALL_RESULT".to_string()));
        match resume_sink.events().last().unwrap() {
            AguiEvent::RunFinished { outcome, .. } => {
                assert!(matches!(outcome, RunOutcome::Success));
            }
            _ => panic!("expected success after resume"),
        }
        assert_eq!(store.status(), "idle");
        assert_eq!(store.tool_status("del_1").as_deref(), Some("success"));
    }

    #[tokio::test]
    async fn subagent_denied_resume_still_finishes() {
        let store = FakeStore::with_user("delegate please");
        let sink = VecSink::default();
        let provider = DelegatingProvider { tool: "worker".to_string() };
        let reg = worker_registry();

        run_turn_with_subagents(
            &store, &FakeExec, &provider, &sink, &params(false), &AllowAll, &reg,
        )
        .await
        .unwrap();
        assert_eq!(store.status(), "responding");

        // Deny the subagent's write; the child records the denial and finishes,
        // and the parent run completes.
        let resume_sink = VecSink::default();
        let item = ResumeItem {
            interrupt_id: "del_1".to_string(),
            payload: ResumePayload { approved: false },
        };
        resume_with_subagents(
            &store, &FakeExec, &provider, &resume_sink, &params(false), &AllowAll, &reg, &item,
        )
        .await
        .unwrap();

        match resume_sink.events().last().unwrap() {
            AguiEvent::RunFinished { outcome, .. } => {
                assert!(matches!(outcome, RunOutcome::Success));
            }
            _ => panic!("expected success after denied resume"),
        }
        assert_eq!(store.status(), "idle");
        assert_eq!(store.tool_status("del_1").as_deref(), Some("success"));
    }
}
