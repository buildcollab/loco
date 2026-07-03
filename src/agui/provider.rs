//! # LLM provider abstraction
//!
//! [`Provider`] is the seam between the reusable run-loop
//! ([`crate::agui::runtime`]) and a concrete LLM backend. A turn is either
//! *non-streaming* ([`Provider::run_turn`], for one-shot callers like title /
//! summary generation) or *streaming* ([`Provider::stream_turn`], which emits
//! [`AgentDelta`]s into a channel while assembling the same [`TurnOutcome`]).
//!
//! Two implementations ship:
//! - [`RigProvider`] — an OpenAI-compatible client aimed at
//!   [OpenRouter](https://openrouter.ai) (direct `reqwest` SSE; no rig
//!   dependency is exposed on the public surface).
//! - [`StubProvider`] — deterministic, network-free, for tests and local dev.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Duration;
use tracing::{debug, warn};

use crate::{Error, Result};

/// A single incremental item produced while streaming a turn.
#[derive(Debug, Clone)]
pub enum AgentDelta {
    /// A chunk of assistant text.
    TextDelta(String),
    /// A tool call has started at stream position `index`.
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// A chunk of the JSON arguments for the tool call at `index`.
    ToolCallArgsDelta { index: usize, delta: String },
    /// The tool call at `index` is complete.
    ToolCallEnd { index: usize },
    /// Token usage for the turn.
    Usage(Usage),
    /// The turn is finished (with the provider's finish reason).
    Done { finish_reason: String },
}

/// Token accounting for a turn.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
}

impl Usage {
    /// Accumulate another usage into this one.
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_tokens += other.cached_tokens;
    }
}

/// Whether a tool only reads (safe to auto-run) or writes (may require
/// human approval before execution).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Read,
    Write,
}

/// A tool the model may call, in JSON-schema form.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub kind: ToolKind,
}

/// A concrete tool invocation requested by the model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallReq {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// A single message in the conversation history sent to the provider.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCallReq>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Convenience constructor for a plain-text message.
    #[must_use]
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Convenience constructor for a `tool` result message.
    #[must_use]
    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }
}

/// The assembled result of a turn.
#[derive(Debug, Clone)]
pub enum TurnOutcome {
    /// The model produced a final text answer.
    Final { text: String, usage: Usage },
    /// The model requested one or more tool calls (with any text produced
    /// before them in `partial_text`).
    Tools {
        calls: Vec<ToolCallReq>,
        usage: Usage,
        partial_text: String,
    },
}

/// Abstraction over an LLM backend.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// The model identifier this provider talks to.
    fn model_id(&self) -> String;

    /// A short label for the provider (persisted alongside messages). Defaults
    /// to `"llm"`; implementations should override with something meaningful.
    fn provider_name(&self) -> String {
        "llm".to_string()
    }

    /// Non-streaming one-shot turn.
    ///
    /// # Errors
    /// Provider/transport failures map to [`Error::Message`].
    async fn run_turn(
        &self,
        system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> Result<TurnOutcome>;

    /// Streaming turn: emit deltas into `tx` **and** return the assembled
    /// outcome. A `tx.send().await` error means the receiver was dropped
    /// (client gone) — return `Err` immediately; that is the abort signal.
    ///
    /// # Errors
    /// Provider/transport failures (and a dropped `tx`) map to [`Error::Message`].
    async fn stream_turn(
        &self,
        system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
        tx: &tokio::sync::mpsc::Sender<AgentDelta>,
    ) -> Result<TurnOutcome>;
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response mapping (shared by RigProvider)
// ---------------------------------------------------------------------------

/// Default OpenRouter base URL.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

fn tools_to_json(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect()
}

fn messages_to_json(system: &str, history: &[ChatMessage], cache_system: bool) -> Vec<Value> {
    let mut out = Vec::with_capacity(history.len() + 1);
    if !system.is_empty() {
        // With `cache_system`, mark the (large, stable) system prompt as a cache
        // breakpoint so Anthropic models via OpenRouter reuse it across turns.
        // OpenAI-family models cache automatically and ignore the annotation.
        if cache_system {
            out.push(json!({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system,
                    "cache_control": { "type": "ephemeral" },
                }],
            }));
        } else {
            out.push(json!({ "role": "system", "content": system }));
        }
    }
    for m in history {
        if m.role == "tool" {
            out.push(json!({
                "role": "tool",
                "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
                "content": m.content,
            }));
        } else if !m.tool_calls.is_empty() {
            let calls: Vec<Value> = m
                .tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": c.arguments.to_string(),
                        }
                    })
                })
                .collect();
            out.push(json!({
                "role": m.role,
                "content": m.content,
                "tool_calls": calls,
            }));
        } else {
            out.push(json!({ "role": m.role, "content": m.content }));
        }
    }
    out
}

/// Incrementally assembles OpenAI-style streaming chunks into deltas + a final
/// [`TurnOutcome`]. Extracted so it can be unit-tested against captured
/// fixtures without a network.
#[derive(Default)]
pub struct StreamAssembler {
    text: String,
    reasoning: String,
    tool_calls: BTreeMap<usize, PartialToolCall>,
    usage: Usage,
    finish_reason: Option<String>,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    args: String,
    started: bool,
    ended: bool,
}

impl StreamAssembler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one parsed chunk object, returning any deltas it produced.
    pub fn ingest(&mut self, chunk: &Value) -> Vec<AgentDelta> {
        let mut deltas = Vec::new();

        if let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) {
            let delta = choice.get("delta").cloned().unwrap_or(Value::Null);

            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    self.text.push_str(content);
                    deltas.push(AgentDelta::TextDelta(content.to_string()));
                }
            }

            // Some models (e.g. `nvidia/nemotron` behind `openrouter/free`) place
            // their prose in an out-of-band `reasoning` field and leave `content`
            // empty. Accumulate it so it isn't dropped; it is only used as the
            // assistant text if `content` ends up empty (see `into_outcome`). It is
            // deliberately not forwarded as a live `TextDelta` so genuine
            // reasoning models that also fill `content` don't double-emit.
            let reasoning = extract_reasoning(&delta);
            if !reasoning.is_empty() {
                self.reasoning.push_str(reasoning);
            }

            if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc in tcs {
                    let index =
                        usize::try_from(tc.get("index").and_then(Value::as_u64).unwrap_or(0))
                            .unwrap_or(0);
                    let entry = self.tool_calls.entry(index).or_default();

                    if let Some(id) = tc.get("id").and_then(Value::as_str) {
                        if !id.is_empty() {
                            entry.id = id.to_string();
                        }
                    }
                    let func = tc.get("function");
                    if let Some(name) = func.and_then(|f| f.get("name")).and_then(Value::as_str) {
                        if !name.is_empty() {
                            entry.name = name.to_string();
                        }
                    }
                    if !entry.started && !entry.id.is_empty() {
                        entry.started = true;
                        deltas.push(AgentDelta::ToolCallStart {
                            index,
                            id: entry.id.clone(),
                            name: entry.name.clone(),
                        });
                    }
                    if let Some(args) = func
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        if !args.is_empty() {
                            entry.args.push_str(args);
                            deltas.push(AgentDelta::ToolCallArgsDelta {
                                index,
                                delta: args.to_string(),
                            });
                        }
                    }
                }
            }

            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(fr.to_string());
                // Close any tool calls that were opened.
                for (index, entry) in &mut self.tool_calls {
                    if entry.started && !entry.ended {
                        entry.ended = true;
                        deltas.push(AgentDelta::ToolCallEnd { index: *index });
                    }
                }
            }
        }

        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = parse_usage(usage);
            deltas.push(AgentDelta::Usage(self.usage.clone()));
        }

        deltas
    }

    /// Emit the terminal `Done` delta (call once the stream reports `[DONE]`).
    pub fn done_delta(&self) -> AgentDelta {
        AgentDelta::Done {
            finish_reason: self
                .finish_reason
                .clone()
                .unwrap_or_else(|| "stop".to_string()),
        }
    }

    /// Consume the assembler into a [`TurnOutcome`].
    #[must_use]
    pub fn into_outcome(self) -> TurnOutcome {
        let Self {
            text,
            reasoning,
            tool_calls,
            usage,
            ..
        } = self;
        // Fall back to `reasoning` when the visible `content` was empty so the
        // assistant message isn't blank for reasoning-in-`reasoning` models.
        let text = if text.is_empty() { reasoning } else { text };
        if tool_calls.is_empty() {
            TurnOutcome::Final { text, usage }
        } else {
            let calls = tool_calls
                .into_values()
                .map(|p| ToolCallReq {
                    id: p.id,
                    name: p.name,
                    arguments: parse_args(&p.args),
                })
                .collect();
            TurnOutcome::Tools {
                calls,
                usage,
                partial_text: text,
            }
        }
    }
}

/// Extract a model's out-of-band reasoning text from a streaming `delta` or a
/// non-streaming `message` object. OpenRouter surfaces it as `reasoning`; some
/// OpenAI-compatible providers (e.g. DeepSeek) use `reasoning_content`. Returns
/// `""` when neither is present.
fn extract_reasoning(v: &Value) -> &str {
    v.get("reasoning")
        .and_then(Value::as_str)
        .or_else(|| v.get("reasoning_content").and_then(Value::as_str))
        .unwrap_or_default()
}

fn parse_args(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return json!({});
    }
    serde_json::from_str(raw).unwrap_or_else(|_| json!({ "_raw": raw }))
}

fn parse_usage(v: &Value) -> Usage {
    let cached = v
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    Usage {
        input_tokens: v.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0),
        output_tokens: v
            .get("completion_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(0),
        cached_tokens: cached,
    }
}

/// Forwarding impl so `Box<dyn Provider>` is itself a `Sized` [`Provider`],
/// letting a heterogeneous registry (e.g. subagents) hold boxed providers and
/// still satisfy `run_turn`'s `Sized` generic bounds.
#[async_trait::async_trait]
impl<T: ?Sized + Provider> Provider for Box<T> {
    fn model_id(&self) -> String {
        (**self).model_id()
    }
    fn provider_name(&self) -> String {
        (**self).provider_name()
    }
    async fn run_turn(
        &self,
        system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> Result<TurnOutcome> {
        (**self).run_turn(system, history, tools).await
    }
    async fn stream_turn(
        &self,
        system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
        tx: &tokio::sync::mpsc::Sender<AgentDelta>,
    ) -> Result<TurnOutcome> {
        (**self).stream_turn(system, history, tools, tx).await
    }
}

// ---------------------------------------------------------------------------
// RigProvider — OpenAI-compatible client (OpenRouter by default)
// ---------------------------------------------------------------------------

/// OpenAI-compatible provider, aimed at OpenRouter by default.
///
/// Despite the name (kept stable for consumers), this is implemented as a
/// direct `reqwest` client against `{base_url}/chat/completions` — streaming
/// via SSE with `stream_options.include_usage`. This gives full control over
/// streaming tool-call and usage handling.
///
/// Note: the `rig` (`rig-core`) crate is **not** used or depended upon anywhere
/// in this project; the `Rig` in the name is purely historical. This is a
/// self-contained `reqwest`-based client with no external agent framework.
pub struct RigProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
    /// Tunables (headers, sampling params, caching, resiliency).
    config: RigConfig,
}

/// Optional tunables for [`RigProvider`]. Defaults leave the request body and
/// headers exactly as before this struct existed.
#[derive(Debug, Clone)]
pub struct RigConfig {
    /// OpenRouter `HTTP-Referer` attribution header.
    pub referer: Option<String>,
    /// OpenRouter `X-Title` attribution header.
    pub title: Option<String>,
    /// Sampling temperature (omitted when `None`).
    pub temperature: Option<f64>,
    /// Max output tokens (omitted when `None`).
    pub max_tokens: Option<i64>,
    /// Nucleus sampling `top_p` (omitted when `None`).
    pub top_p: Option<f64>,
    /// Mark the system prompt as a cache breakpoint (Anthropic via OpenRouter).
    pub cache_system: bool,
    /// Max retry attempts for transient failures (429 / 5xx / connect).
    pub max_retries: usize,
    /// Idle timeout between streamed chunks before aborting a stalled stream.
    pub idle_timeout: Duration,
    /// Overall timeout for a non-streaming request.
    pub request_timeout: Duration,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
}

impl Default for RigConfig {
    fn default() -> Self {
        Self {
            referer: None,
            title: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            cache_system: false,
            max_retries: 2,
            idle_timeout: Duration::from_secs(60),
            request_timeout: Duration::from_secs(120),
            connect_timeout: Duration::from_secs(10),
        }
    }
}

impl RigProvider {
    /// Build a provider with default tunables. `base_url` defaults to
    /// [`OPENROUTER_BASE_URL`] when `None`.
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::with_config(api_key, base_url, model, RigConfig::default())
    }

    /// Build a provider with explicit [`RigConfig`] tunables.
    #[must_use]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: impl Into<String>,
        config: RigConfig,
    ) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .build()
            .unwrap_or_default();
        Self {
            api_key: api_key.into(),
            base_url: base_url.unwrap_or_else(|| OPENROUTER_BASE_URL.to_string()),
            model: model.into(),
            client,
            config,
        }
    }

    /// Set OpenRouter attribution headers (builder style).
    #[must_use]
    pub fn with_headers(
        mut self,
        referer: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        self.config.referer = Some(referer.into());
        self.config.title = Some(title.into());
        self
    }

    /// Set sampling parameters (builder style); pass `None` to leave a param off.
    #[must_use]
    pub fn with_sampling(
        mut self,
        temperature: Option<f64>,
        max_tokens: Option<i64>,
        top_p: Option<f64>,
    ) -> Self {
        self.config.temperature = temperature;
        self.config.max_tokens = max_tokens;
        self.config.top_p = top_p;
        self
    }

    /// Enable a system-prompt cache breakpoint (builder style).
    #[must_use]
    pub fn with_cache_system(mut self, cache_system: bool) -> Self {
        self.config.cache_system = cache_system;
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Base request body shared by streaming and non-streaming turns.
    fn build_body(&self, system: &str, history: &[ChatMessage], tools: &[ToolSpec]) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages_to_json(system, history, self.config.cache_system),
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools_to_json(tools));
        }
        if let Some(t) = self.config.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(m) = self.config.max_tokens {
            body["max_tokens"] = json!(m);
        }
        if let Some(p) = self.config.top_p {
            body["top_p"] = json!(p);
        }
        body
    }

    /// Attach auth + optional OpenRouter attribution headers.
    fn prepare(&self, body: &Value, timeout: Option<Duration>) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .json(body);
        if let Some(referer) = &self.config.referer {
            req = req.header("HTTP-Referer", referer);
        }
        if let Some(title) = &self.config.title {
            req = req.header("X-Title", title);
        }
        if let Some(t) = timeout {
            req = req.timeout(t);
        }
        req
    }

    /// Send the request, retrying transient failures (429 / 5xx / connect /
    /// timeout) with exponential backoff. Returns the successful (2xx) response;
    /// safe for streaming because retries happen before any body is read.
    async fn send_with_retry(
        &self,
        body: &Value,
        timeout: Option<Duration>,
        streaming: bool,
    ) -> Result<reqwest::Response> {
        let mut attempt: u32 = 0;
        loop {
            debug!(
                target: "loco_rs::agui",
                model = %self.model, streaming, attempt, "provider request"
            );
            match self.prepare(body, timeout).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = parse_retry_after(&resp);
                    let text = resp.text().await.unwrap_or_default();
                    if is_retryable_status(status) && attempt < self.config.max_retries as u32 {
                        let delay = retry_after.unwrap_or_else(|| backoff_delay(attempt));
                        warn!(
                            target: "loco_rs::agui",
                            %status, attempt, backoff_ms = delay.as_millis() as u64,
                            "provider retry (status)"
                        );
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(Error::Message(format!(
                        "provider returned {status}: {text}"
                    )));
                }
                Err(e) => {
                    if is_retryable_err(&e) && attempt < self.config.max_retries as u32 {
                        let delay = backoff_delay(attempt);
                        warn!(
                            target: "loco_rs::agui",
                            error = %e, attempt, backoff_ms = delay.as_millis() as u64,
                            "provider retry (transport)"
                        );
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(Error::Message(format!("provider request failed: {e}")));
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl Provider for RigProvider {
    fn model_id(&self) -> String {
        self.model.clone()
    }

    fn provider_name(&self) -> String {
        "openrouter".to_string()
    }

    async fn run_turn(
        &self,
        system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> Result<TurnOutcome> {
        let body = self.build_body(system, history, tools);
        let resp = self
            .send_with_retry(&body, Some(self.config.request_timeout), false)
            .await?;

        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::Message(format!("provider response decode failed: {e}")))?;
        let outcome = parse_non_streaming(&v)?;
        debug!(target: "loco_rs::agui", usage = ?outcome_usage(&outcome), "provider turn done");
        Ok(outcome)
    }

    async fn stream_turn(
        &self,
        system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
        tx: &tokio::sync::mpsc::Sender<AgentDelta>,
    ) -> Result<TurnOutcome> {
        use futures_util::StreamExt;

        let mut body = self.build_body(system, history, tools);
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });

        // No overall timeout on a stream; the connect timeout + per-chunk idle
        // timeout below bound it. Retries happen before the first byte is read.
        let resp = self.send_with_retry(&body, None, true).await?;

        let mut assembler = StreamAssembler::new();
        let mut stream = resp.bytes_stream();
        // Buffer raw bytes (not lossy-decoded strings) so a multi-byte codepoint
        // split across two chunks is never corrupted; decode on line boundaries.
        let mut buf: Vec<u8> = Vec::new();

        loop {
            let next = tokio::time::timeout(self.config.idle_timeout, stream.next()).await;
            let chunk = match next {
                Ok(Some(chunk)) => {
                    chunk.map_err(|e| Error::Message(format!("provider stream error: {e}")))?
                }
                Ok(None) => break, // stream ended
                Err(_) => {
                    return Err(Error::Message(format!(
                        "provider stream idle for {}s",
                        self.config.idle_timeout.as_secs()
                    )))
                }
            };
            buf.extend_from_slice(&chunk);

            // Process complete lines; keep any trailing partial line in `buf`.
            for line in drain_lines(&mut buf) {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    let _ = tx.send(assembler.done_delta()).await;
                    let outcome = assembler.into_outcome();
                    debug!(target: "loco_rs::agui", usage = ?outcome_usage(&outcome), "provider stream done");
                    return Ok(outcome);
                }
                let Ok(json) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                for delta in assembler.ingest(&json) {
                    // A send error means the receiver was dropped — abort.
                    tx.send(delta).await.map_err(|_| {
                        Error::string("event receiver dropped (client disconnected)")
                    })?;
                }
            }
        }

        // Stream ended without an explicit [DONE].
        let _ = tx.send(assembler.done_delta()).await;
        let outcome = assembler.into_outcome();
        debug!(target: "loco_rs::agui", usage = ?outcome_usage(&outcome), "provider stream done");
        Ok(outcome)
    }
}

/// Drain complete `\n`-terminated lines from a raw byte buffer, decoding each
/// full line as UTF-8. Decoding a *complete* line (only after its trailing
/// newline has arrived) means a multi-byte codepoint split across two network
/// chunks is never corrupted — the fix for the prior per-chunk `from_utf8_lossy`.
/// Any trailing partial line stays buffered for the next chunk.
fn drain_lines(buf: &mut Vec<u8>) -> Vec<String> {
    let mut lines = Vec::new();
    while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
        let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
        let line = String::from_utf8_lossy(&line_bytes[..line_bytes.len() - 1]);
        lines.push(line.trim().to_string());
    }
    lines
}

/// Whether an HTTP status is worth retrying (rate limit or server error).
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

/// Whether a transport error is transient (connect/timeout/request-send).
fn is_retryable_err(e: &reqwest::Error) -> bool {
    e.is_connect() || e.is_timeout() || e.is_request()
}

/// Exponential backoff with light jitter: ~0.5s, 1s, 2s, … capped at 8s.
fn backoff_delay(attempt: u32) -> Duration {
    let base_ms = 500u64.saturating_mul(1u64 << attempt.min(4));
    let capped = base_ms.min(8_000);
    // Cheap deterministic-ish jitter (0..128ms) from the system clock nanos.
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()) % 128)
        .unwrap_or(0);
    Duration::from_millis(capped + jitter)
}

/// Parse a `Retry-After` header (delta-seconds form) into a duration.
fn parse_retry_after(resp: &reqwest::Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Extract the [`Usage`] from an outcome (for logging only).
fn outcome_usage(outcome: &TurnOutcome) -> &Usage {
    match outcome {
        TurnOutcome::Final { usage, .. } | TurnOutcome::Tools { usage, .. } => usage,
    }
}

fn parse_non_streaming(v: &Value) -> Result<TurnOutcome> {
    let usage = v
        .get("usage")
        .filter(|u| !u.is_null())
        .map(parse_usage)
        .unwrap_or_default();

    let message = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .ok_or_else(|| Error::string("provider response missing choices[0].message"))?;

    let mut text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Reasoning-in-`reasoning` models (e.g. `openrouter/free` -> nemotron) leave
    // `content` empty; fall back to their reasoning text so nothing is lost.
    if text.is_empty() {
        text = extract_reasoning(message).to_string();
    }

    if let Some(tcs) = message.get("tool_calls").and_then(Value::as_array) {
        if !tcs.is_empty() {
            let calls = tcs
                .iter()
                .map(|tc| {
                    let id = tc
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let func = tc.get("function");
                    let name = func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let args = func
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    ToolCallReq {
                        id,
                        name,
                        arguments: parse_args(args),
                    }
                })
                .collect();
            return Ok(TurnOutcome::Tools {
                calls,
                usage,
                partial_text: text,
            });
        }
    }

    Ok(TurnOutcome::Final { text, usage })
}

// ---------------------------------------------------------------------------
// StubProvider — deterministic, network-free
// ---------------------------------------------------------------------------

/// A deterministic, sleep-free provider for tests and local development.
///
/// If the last user message reads like a write intent and a [`ToolKind::Write`]
/// spec is available, it emits a single tool-call sequence for that tool.
/// Otherwise it chunks a canned reply into a few text deltas. No app tool names
/// are hardcoded.
#[derive(Debug, Clone, Default)]
pub struct StubProvider {
    /// Optional canned reply; defaults to a short generic message.
    pub reply: Option<String>,
}

impl StubProvider {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_reply(reply: impl Into<String>) -> Self {
        Self {
            reply: Some(reply.into()),
        }
    }

    fn canned(&self) -> String {
        self.reply
            .clone()
            .unwrap_or_else(|| "Hello from the stub provider.".to_string())
    }

    fn plan(&self, history: &[ChatMessage], tools: &[ToolSpec]) -> Plan {
        // Only consider a tool call when the conversation is waiting on the
        // model right after a *user* turn. Once a tool result has been appended
        // (last role is `tool`/`assistant`), fall through to a text answer so
        // the run-loop terminates instead of re-requesting the same tool.
        let last_is_user = history.last().map(|m| m.role == "user").unwrap_or(false);

        let last_user = history
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.to_lowercase())
            .unwrap_or_default();

        let write_intent = ["create", "write", "update", "delete", "add", "set", "make"]
            .iter()
            .any(|kw| last_user.contains(kw));

        if last_is_user && write_intent {
            if let Some(spec) = tools.iter().find(|t| t.kind == ToolKind::Write) {
                return Plan::Tool {
                    id: format!("call_stub_{}", spec.name),
                    name: spec.name.clone(),
                    args: json!({ "request": last_user }),
                };
            }
        }
        Plan::Text(self.canned())
    }
}

enum Plan {
    Text(String),
    Tool { id: String, name: String, args: Value },
}

#[async_trait::async_trait]
impl Provider for StubProvider {
    fn model_id(&self) -> String {
        "stub-model".to_string()
    }

    fn provider_name(&self) -> String {
        "stub".to_string()
    }

    async fn run_turn(
        &self,
        _system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> Result<TurnOutcome> {
        Ok(match self.plan(history, tools) {
            Plan::Text(text) => TurnOutcome::Final {
                text,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                },
            },
            Plan::Tool { id, name, args } => TurnOutcome::Tools {
                calls: vec![ToolCallReq {
                    id,
                    name,
                    arguments: args,
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                },
                partial_text: String::new(),
            },
        })
    }

    async fn stream_turn(
        &self,
        _system: &str,
        history: &[ChatMessage],
        tools: &[ToolSpec],
        tx: &tokio::sync::mpsc::Sender<AgentDelta>,
    ) -> Result<TurnOutcome> {
        let usage = Usage {
            input_tokens: 1,
            output_tokens: 1,
            cached_tokens: 0,
        };
        let send = |d: AgentDelta| async {
            tx.send(d)
                .await
                .map_err(|_| Error::string("event receiver dropped (client disconnected)"))
        };

        match self.plan(history, tools) {
            Plan::Text(text) => {
                // Chunk the reply into a few deltas.
                for piece in chunk_words(&text, 3) {
                    send(AgentDelta::TextDelta(piece)).await?;
                }
                send(AgentDelta::Usage(usage.clone())).await?;
                send(AgentDelta::Done {
                    finish_reason: "stop".to_string(),
                })
                .await?;
                Ok(TurnOutcome::Final { text, usage })
            }
            Plan::Tool { id, name, args } => {
                let args_str = args.to_string();
                send(AgentDelta::ToolCallStart {
                    index: 0,
                    id: id.clone(),
                    name: name.clone(),
                })
                .await?;
                send(AgentDelta::ToolCallArgsDelta {
                    index: 0,
                    delta: args_str,
                })
                .await?;
                send(AgentDelta::ToolCallEnd { index: 0 }).await?;
                send(AgentDelta::Usage(usage.clone())).await?;
                send(AgentDelta::Done {
                    finish_reason: "tool_calls".to_string(),
                })
                .await?;
                Ok(TurnOutcome::Tools {
                    calls: vec![ToolCallReq {
                        id,
                        name,
                        arguments: args,
                    }],
                    usage,
                    partial_text: String::new(),
                })
            }
        }
    }
}

/// Split `text` into chunks of at most `n` whitespace-separated words,
/// preserving a trailing space between chunks so re-concatenation is lossless
/// up to internal whitespace normalization.
fn chunk_words(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![text.to_string()];
    }
    words
        .chunks(n.max(1))
        .map(|c| c.join(" "))
        .collect()
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn write_spec() -> ToolSpec {
        ToolSpec {
            name: "save_note".to_string(),
            description: "Save a note".to_string(),
            parameters: json!({"type": "object"}),
            kind: ToolKind::Write,
        }
    }

    async fn collect(
        provider: &StubProvider,
        history: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> (Vec<AgentDelta>, TurnOutcome) {
        let (tx, mut rx) = mpsc::channel(64);
        let history = history.to_vec();
        let tools = tools.to_vec();
        let p = provider.clone();
        let handle = tokio::spawn(async move { p.stream_turn("", &history, &tools, &tx).await });
        let mut deltas = Vec::new();
        while let Some(d) = rx.recv().await {
            deltas.push(d);
        }
        let outcome = handle.await.unwrap().unwrap();
        (deltas, outcome)
    }

    #[tokio::test]
    async fn stub_text_path() {
        let stub = StubProvider::with_reply("one two three four five");
        let history = vec![ChatMessage::text("user", "just chatting")];
        let (deltas, outcome) = collect(&stub, &history, &[]).await;

        assert!(matches!(deltas.first(), Some(AgentDelta::TextDelta(_))));
        assert!(deltas
            .iter()
            .any(|d| matches!(d, AgentDelta::Usage(_))));
        assert!(matches!(deltas.last(), Some(AgentDelta::Done { .. })));
        match outcome {
            TurnOutcome::Final { text, .. } => assert!(text.contains("one")),
            TurnOutcome::Tools { .. } => panic!("expected Final"),
        }
    }

    #[tokio::test]
    async fn stub_tool_path() {
        let stub = StubProvider::new();
        let history = vec![ChatMessage::text("user", "please create a note")];
        let tools = vec![write_spec()];
        let (deltas, outcome) = collect(&stub, &history, &tools).await;

        assert!(matches!(
            deltas.first(),
            Some(AgentDelta::ToolCallStart { .. })
        ));
        assert!(deltas
            .iter()
            .any(|d| matches!(d, AgentDelta::ToolCallArgsDelta { .. })));
        assert!(deltas
            .iter()
            .any(|d| matches!(d, AgentDelta::ToolCallEnd { .. })));
        match outcome {
            TurnOutcome::Tools { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "save_note");
            }
            TurnOutcome::Final { .. } => panic!("expected Tools"),
        }
    }

    #[tokio::test]
    async fn stub_run_turn_parity() {
        let stub = StubProvider::new();
        // Text path
        let text_hist = vec![ChatMessage::text("user", "hello there")];
        assert!(matches!(
            stub.run_turn("", &text_hist, &[]).await.unwrap(),
            TurnOutcome::Final { .. }
        ));
        // Tool path
        let tool_hist = vec![ChatMessage::text("user", "update the record")];
        assert!(matches!(
            stub.run_turn("", &tool_hist, &[write_spec()]).await.unwrap(),
            TurnOutcome::Tools { .. }
        ));
    }

    #[test]
    fn assembler_text_and_usage() {
        let mut a = StreamAssembler::new();
        a.ingest(&json!({"choices":[{"delta":{"content":"Hel"}}]}));
        a.ingest(&json!({"choices":[{"delta":{"content":"lo"}}]}));
        a.ingest(&json!({"choices":[{"delta":{},"finish_reason":"stop"}]}));
        a.ingest(&json!({"usage":{"prompt_tokens":10,"completion_tokens":2}}));
        match a.into_outcome() {
            TurnOutcome::Final { text, usage } => {
                assert_eq!(text, "Hello");
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 2);
            }
            TurnOutcome::Tools { .. } => panic!("expected Final"),
        }
    }

    #[test]
    fn assembler_tool_args_split_across_chunks() {
        let mut a = StreamAssembler::new();
        // First chunk: id + name
        let d0 = a.ingest(&json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"name":"search","arguments":"{\"q\":"}}
        ]}}]}));
        assert!(d0
            .iter()
            .any(|d| matches!(d, AgentDelta::ToolCallStart { .. })));
        // Second chunk: more args
        a.ingest(&json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"\"rust\"}"}}
        ]}}]}));
        // finish
        let dend = a.ingest(&json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}));
        assert!(dend
            .iter()
            .any(|d| matches!(d, AgentDelta::ToolCallEnd { .. })));
        a.ingest(&json!({"usage":{"prompt_tokens":5,"completion_tokens":7}}));

        match a.into_outcome() {
            TurnOutcome::Tools { calls, usage, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["q"], "rust");
                assert_eq!(usage.output_tokens, 7);
            }
            TurnOutcome::Final { .. } => panic!("expected Tools"),
        }
    }

    #[test]
    fn assembler_openrouter_free_reasoning_with_tool_call() {
        // Shape observed from `openrouter/free` (-> `nvidia/nemotron`): the prose
        // arrives in `reasoning`, `content` stays empty, and the tool call is
        // streamed normally. The tool call must survive and the reasoning prose
        // must be surfaced rather than dropped.
        let mut a = StreamAssembler::new();
        a.ingest(&json!({"choices":[{"delta":{"reasoning":"I should list the tasks. "}}]}));
        a.ingest(&json!({"choices":[{"delta":{"reasoning":"Calling list_tasks."}}]}));
        a.ingest(&json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"name":"list_tasks","arguments":"{}"}}
        ]}}]}));
        a.ingest(&json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}));
        a.ingest(&json!({"usage":{"prompt_tokens":20,"completion_tokens":5}}));

        match a.into_outcome() {
            TurnOutcome::Tools {
                calls,
                partial_text,
                ..
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "list_tasks");
                assert_eq!(calls[0].arguments, json!({}));
                assert!(partial_text.contains("list the tasks"));
            }
            TurnOutcome::Final { .. } => panic!("expected Tools"),
        }
    }

    #[test]
    fn assembler_reasoning_fallback_when_content_empty() {
        let mut a = StreamAssembler::new();
        a.ingest(&json!({"choices":[{"delta":{"reasoning":"Hello "}}]}));
        a.ingest(&json!({"choices":[{"delta":{"reasoning":"there."}}]}));
        a.ingest(&json!({"choices":[{"delta":{},"finish_reason":"stop"}]}));
        match a.into_outcome() {
            TurnOutcome::Final { text, .. } => assert_eq!(text, "Hello there."),
            TurnOutcome::Tools { .. } => panic!("expected Final"),
        }
    }

    #[test]
    fn assembler_content_preferred_over_reasoning() {
        // A genuine reasoning model that fills both fields must keep `content` as
        // the visible answer and not leak its reasoning.
        let mut a = StreamAssembler::new();
        a.ingest(&json!({"choices":[{"delta":{"reasoning":"thinking..."}}]}));
        a.ingest(&json!({"choices":[{"delta":{"content":"Answer"}}]}));
        a.ingest(&json!({"choices":[{"delta":{},"finish_reason":"stop"}]}));
        match a.into_outcome() {
            TurnOutcome::Final { text, .. } => assert_eq!(text, "Answer"),
            TurnOutcome::Tools { .. } => panic!("expected Final"),
        }
    }

    #[test]
    fn non_streaming_openrouter_free_reasoning_with_tool_call() {
        // Non-streaming counterpart of the `openrouter/free` tool-call shape.
        let v = json!({
            "choices": [{"message": {
                "content": "",
                "reasoning": "I'll list the tasks now.",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "list_tasks", "arguments": "{}"}}
                ]
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5}
        });
        match parse_non_streaming(&v).unwrap() {
            TurnOutcome::Tools {
                calls,
                partial_text,
                ..
            } => {
                assert_eq!(calls[0].name, "list_tasks");
                assert!(partial_text.contains("list the tasks"));
            }
            TurnOutcome::Final { .. } => panic!("expected Tools"),
        }
    }

    #[test]
    fn non_streaming_reasoning_content_fallback() {
        // DeepSeek-style `reasoning_content` field, empty `content`, no tools.
        let v = json!({
            "choices": [{"message": {"content": "", "reasoning_content": "Fallback answer."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        match parse_non_streaming(&v).unwrap() {
            TurnOutcome::Final { text, .. } => assert_eq!(text, "Fallback answer."),
            TurnOutcome::Tools { .. } => panic!("expected Final"),
        }
    }

    #[test]
    fn non_streaming_decode_tools() {
        let v = json!({
            "choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}
            ]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}
        });
        match parse_non_streaming(&v).unwrap() {
            TurnOutcome::Tools { calls, .. } => {
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["q"], "x");
            }
            TurnOutcome::Final { .. } => panic!("expected Tools"),
        }
    }

    // ----- Stage 1: resiliency / caching / config ---------------------------

    #[test]
    fn utf8_codepoint_split_across_chunks_is_intact() {
        // "café ☕" as SSE data, with the multi-byte 'é' (0xC3 0xA9) and the
        // emoji split across two buffer appends. The old per-chunk
        // `from_utf8_lossy` would corrupt these; `drain_lines` must not.
        let full = "data: caf\u{e9} \u{2615}\n".as_bytes().to_vec();
        let split = full.len() - 3; // slice through a multi-byte sequence
        let mut buf: Vec<u8> = Vec::new();

        buf.extend_from_slice(&full[..split]);
        assert!(drain_lines(&mut buf).is_empty(), "no complete line yet");
        buf.extend_from_slice(&full[split..]);
        let lines = drain_lines(&mut buf);

        assert_eq!(lines, vec!["data: caf\u{e9} \u{2615}".to_string()]);
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_lines_keeps_partial_trailing_line() {
        let mut buf = b"data: a\ndata: b".to_vec();
        let lines = drain_lines(&mut buf);
        assert_eq!(lines, vec!["data: a".to_string()]);
        // The partial "data: b" (no newline) stays buffered.
        assert_eq!(buf, b"data: b");
    }

    #[test]
    fn retryable_status_classification() {
        use reqwest::StatusCode;
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_status(StatusCode::OK));
    }

    #[test]
    fn backoff_grows_and_is_capped() {
        let d0 = backoff_delay(0).as_millis();
        let d1 = backoff_delay(1).as_millis();
        // Base grows ~2x per attempt (jitter is < 128ms, base starts at 500ms).
        assert!(d0 >= 500 && d0 < 700);
        assert!(d1 >= 1000 && d1 < 1200);
        // Capped at 8s + jitter regardless of attempt.
        assert!(backoff_delay(20).as_millis() <= 8_128);
    }

    #[test]
    fn messages_to_json_cache_breakpoint_opt_in() {
        let hist = vec![ChatMessage::text("user", "hi")];
        // Off: plain string system content.
        let plain = messages_to_json("sys", &hist, false);
        assert_eq!(plain[0]["content"], json!("sys"));
        // On: array content with an ephemeral cache_control breakpoint.
        let cached = messages_to_json("sys", &hist, true);
        assert_eq!(cached[0]["content"][0]["type"], "text");
        assert_eq!(cached[0]["content"][0]["text"], "sys");
        assert_eq!(
            cached[0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn build_body_folds_sampling_and_cache_only_when_set() {
        // Defaults: no sampling params, plain system content.
        let base = RigProvider::new("k", None, "m");
        let b0 = base.build_body("sys", &[], &[]);
        assert!(b0.get("temperature").is_none());
        assert!(b0.get("max_tokens").is_none());
        assert!(b0.get("top_p").is_none());
        assert_eq!(b0["messages"][0]["content"], json!("sys"));

        // Configured: params appear; system prompt is a cache breakpoint.
        let tuned = RigProvider::new("k", None, "m")
            .with_sampling(Some(0.2), Some(256), Some(0.9))
            .with_cache_system(true);
        let b1 = tuned.build_body("sys", &[], &[]);
        assert_eq!(b1["temperature"], json!(0.2));
        assert_eq!(b1["max_tokens"], json!(256));
        assert_eq!(b1["top_p"], json!(0.9));
        assert_eq!(b1["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn with_headers_sets_attribution() {
        let p = RigProvider::new("k", None, "m").with_headers("https://example.com", "MyApp");
        assert_eq!(p.config.referer.as_deref(), Some("https://example.com"));
        assert_eq!(p.config.title.as_deref(), Some("MyApp"));
    }

    /// Live end-to-end verification that `openrouter/free` drives a tool call
    /// through [`RigProvider`]. Ignored by default (needs network + a key); run
    /// it explicitly to reproduce the "does the tool flow work" check:
    ///
    /// ```bash
    /// OPENROUTER_API_KEY=sk-or-... \
    ///   cargo test --features agui -p loco-rs \
    ///   agui::provider::tests::live_openrouter_free_emits_tool_call -- --ignored --nocapture
    /// ```
    ///
    /// It sends a prompt that can only be answered by listing tasks, exposing a
    /// single read tool, and asserts the provider returns a [`TurnOutcome::Tools`]
    /// naming that tool — i.e. the model calls tools and our parser assembles the
    /// call regardless of whether prose landed in `content` or `reasoning`.
    #[tokio::test]
    #[ignore = "hits the live OpenRouter API; requires OPENROUTER_API_KEY"]
    async fn live_openrouter_free_emits_tool_call() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            eprintln!("skipping: OPENROUTER_API_KEY not set");
            return;
        }

        let provider = RigProvider::new(api_key, None, "openrouter/free");
        let tools = vec![ToolSpec {
            name: "list_tasks".to_string(),
            description: "List the user's current tasks.".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
            kind: ToolKind::Read,
        }];
        let history = vec![ChatMessage::text(
            "user",
            "What tasks do I currently have? Use the available tool to find out.",
        )];

        let outcome = provider
            .run_turn(
                "You are a helpful assistant. Use tools when they can answer the question.",
                &history,
                &tools,
            )
            .await
            .expect("provider request should succeed");

        match outcome {
            TurnOutcome::Tools { calls, .. } => {
                assert!(
                    calls.iter().any(|c| c.name == "list_tasks"),
                    "expected a list_tasks call, got: {:?}",
                    calls.iter().map(|c| &c.name).collect::<Vec<_>>()
                );
            }
            TurnOutcome::Final { text, .. } => {
                panic!("expected a tool call, got final text: {text:?}");
            }
        }
    }
}
