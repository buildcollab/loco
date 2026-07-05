+++
title = "AI Agents (AG-UI)"
description = ""
date = 2026-07-04T00:00:00+00:00
updated = 2026-07-04T00:00:00+00:00
draft = false
weight = 7
sort_by = "weight"
template = "docs/page.html"

[extra]
lead = ""
toc = true
top = false
flair =[]
+++

Loco's `agui` module runs streaming LLM agents and exposes them to a frontend
over the [AG-UI protocol](https://docs.ag-ui.com) using Server-Sent Events. It
gives you a resumable, cancellable run loop with typed tools, lifecycle hooks,
subagents, and per-call authorization — plus optional **durable execution** on
the background-worker queue.

The module is generic infrastructure with **zero business logic**: everything
app-specific (which agents exist, their prompts, tools, and models) is declared
in your app; the persistence, provider wiring, run hub, HTTP router, and worker
all live in the framework.

## Enabling

Turn on the `agui` feature (it builds on `with-db`, which is on by default):

```toml
# Cargo.toml
loco-rs = { version = "*", features = ["agui"] }
```

## Generating an agent

```sh
$ cargo loco generate agent support
$ cargo loco db migrate && cargo loco db entities
```

The generator is deliberately thin — it scaffolds only what is specific to
*your* agent:

```text
src/agents/
  mod.rs             # the agent registry
  support/
    mod.rs           # the agent: name, model, system prompt
    tools.rs         # its typed tools
    hooks.rs         # its lifecycle callbacks
src/controllers/
  agents.rs          # a one-line mount of the framework router
migration/
  m<ts>_agents.rs    # the agent tables
```

Everything else — the conversation store, the run hub, the provider factory,
the HTTP handlers, and the durable worker — is **library code** in
`loco_rs::agui`, shared by every agent. Add more agents with the same command;
each new one is just another `src/agents/<name>/` module plus a registry line.

### Declaring an agent

An agent implements the [`Agent`] trait; its `name()` is its id, stored on each
conversation as `agent_id`:

```rust
// src/agents/support/mod.rs
pub struct SupportAgent;

#[async_trait]
impl Agent for SupportAgent {
    fn name(&self) -> &str { "support" }
    fn model(&self) -> String { "anthropic/claude-sonnet-5".to_string() }

    async fn system_prompt(&self, ctx: &AgentCtx<'_>) -> Result<String> {
        // Appends the conversation's mode + attached context items.
        loco_rs::agui::assemble_system(ctx, "You are support, a helpful assistant.").await
    }

    fn tools(&self) -> Tools { tools::tools() }
    fn hooks(&self) -> Arc<dyn AgentHooks> { Arc::new(hooks::SupportHooks) }
}
```

Agents are collected into a registry the controller resolves against:

```rust
// src/agents/mod.rs
pub fn registry() -> AgentRegistry {
    let mut registry = AgentRegistry::new();
    registry.register(support::SupportAgent);
    registry
}
```

## Configuration

Provider credentials, the run-hub backend, and the execution mode are chosen at
config time so they stay out of code:

```yaml
# config/development.yaml
agui:
  provider:
    kind: openrouter        # openrouter | openai | openai_compatible
    api_key: "{{ get_env(name='OPENROUTER_API_KEY', default='') }}"
    # base_url: ...         # required only for openai_compatible
    default_model: anthropic/claude-sonnet-5
  hub:
    kind: in_mem            # in_mem | redis | postgres
  execution:
    kind: inline            # inline | worker
```

- **provider** — any OpenAI-compatible endpoint. `default_model` overrides the
  agent's declared model when set.
- **hub** — where per-run events are buffered and cancellation is coordinated
  (see [Run hub](#run-hub-resumable-cancellable-streams)).
- **execution** — inline or durable worker (see
  [Durable execution](#durable-execution-background-workers)).

## HTTP API

The generated controller mounts the framework router with your registry:

```rust
// src/controllers/agents.rs — the whole file
pub fn routes() -> Routes {
    loco_rs::agui::controller::routes(Arc::new(crate::agents::registry()))
}
```

That gives you, under `/api`:

| Method & path                                | Purpose                                  |
|----------------------------------------------|------------------------------------------|
| `GET  /agents`                               | list declared agents                     |
| `GET  /agents/{agent_id}`                    | one agent                                |
| `GET  /agents/{agent_id}/conversations`      | list conversations                       |
| `POST /agents/{agent_id}/conversations`      | open a conversation                      |
| `GET  /conversations/{pid}/messages`         | message history                          |
| `POST /conversations/{pid}/context`          | attach context to the system prompt      |
| `POST /conversations/{pid}/run`              | start a run (SSE, resumable)             |
| `GET  /conversations/{pid}/stream?since=N`   | resume the live stream                   |
| `POST /conversations/{pid}/cancel`           | stop the active run                      |

Start a run by POSTing an AG-UI `RunAgentInput` body:

```jsonc
POST /api/conversations/{pid}/run
{
  "runId": "optional-client-supplied-id",
  "message": "How do I reset my password?",
  "resume": []            // resume/approve instructions for an interrupted run
}
```

The response is an SSE stream of AG-UI events (`TEXT_MESSAGE_CONTENT`,
`TOOL_CALL_*`, `RUN_FINISHED`, ...).

## Run hub: resumable & cancellable streams

The run is **decoupled from the HTTP connection** by a run hub. The run publishes
events into the hub; the HTTP handler subscribes to the hub and forwards them.
Two properties fall out of this:

- **Resumable** — every event carries a monotonic per-run sequence number
  (surfaced as the SSE `id:`). If the connection drops, the client reconnects
  with `GET /conversations/{pid}/stream?since=<lastEventId>` and replays from
  exactly where it left off. A dropped `fetch` does **not** cancel the run.
- **Cancellable** — `POST /conversations/{pid}/cancel` flips the run's
  cancellation token; the loop stops cooperatively, persisting partial output.

### Backends

| `agui.hub` | Backend            | Scope                                                        |
|------------|--------------------|-------------------------------------------------------------|
| `in_mem`   | `InMemoryRunHub`   | Single process — a per-run buffer + broadcast. The default. |
| `redis`    | `DbRunHub`         | Multi-node — events persist to `agent_events`, cancellation rides on `agent_runs`. |
| `postgres` | `DbRunHub`         | Multi-node — same, over Postgres.                           |

With a DB-backed hub any node can serve a reconnect, because the event log lives
in the database rather than one process's memory.

## Durable execution (background workers)

By default a run is driven on a `tokio::spawn` task inside the web process. That
survives a dropped client connection — but **not** a restart of that process.

Set `agui.execution` to `worker` to hand each run to the
[background-worker queue](@/docs/processing/workers.md) instead. A worker picks
it up and drives it, publishing to the (DB-backed) run hub, while the web node
that took the request keeps streaming to the client. The run is now **durable**:
it survives a web-process restart and can be retried by the queue.

```yaml
# config/production.yaml
workers:
  mode: BackgroundQueue     # required for worker execution
agui:
  hub:
    kind: postgres          # a DB-backed hub is required (redis | postgres)
  execution:
    kind: worker
```

Because worker execution needs a real queue, register the worker in
`connect_workers`, handing it the agent registry (the registry can't be
reconstructed from the serialized job payload, so the running instance carries
it):

```rust
// src/app.rs
async fn connect_workers(ctx: &AppContext, queue: &Queue) -> Result<()> {
    queue
        .register(loco_rs::agui::worker::RunAgentJob::with_registry(
            ctx,
            std::sync::Arc::new(crate::agents::registry()),
        ))
        .await?;
    Ok(())
}
```

Nothing else changes: the same `POST /run` endpoint enqueues the run instead of
spawning it, and the same `GET /stream` / `POST /cancel` endpoints work across
nodes through the shared `agent_events` / `agent_runs` tables.

> **Note:** Worker execution requires `workers.mode: BackgroundQueue` **and** a
> DB-backed `agui.hub` (`redis` or `postgres`) — with an in-memory hub a worker
> on another node could not share its event stream. If `agui.execution` is
> `worker` but the queue is not in `BackgroundQueue` mode, Loco logs a warning
> and falls back to inline execution so runs still work.

### Inline vs. worker

| Property                          | `inline` (default) | `worker`                     |
|-----------------------------------|--------------------|------------------------------|
| Survives a dropped connection     | ✅                 | ✅                           |
| Survives a web-process restart    | ❌                 | ✅                           |
| Retryable by the queue            | ❌                 | ✅                           |
| Requires a queue                  | ❌                 | ✅ (`BackgroundQueue`)       |
| Requires a DB-backed hub          | ❌                 | ✅ (`redis` / `postgres`)    |

## Headless runs (no client attached)

A run does not need an HTTP request or a browser watching the stream. To kick
off an agent from a **task**, the **scheduler**, or another **worker** — e.g. to
generate a report asynchronously — call `start_run`:

```rust
use loco_rs::agui::{start_run, Principal};

let run = start_run(
    &ctx,
    &std::sync::Arc::new(crate::agents::registry()),
    "report_writer",                       // agent id
    "Generate the Q3 sales report",        // the trigger prompt
    Principal::default(),                  // no request principal in the background
    None,                                  // tenancy scope (e.g. Some(json!({ "organization_id": 42 })))
).await?;
```

`start_run` opens a fresh conversation for the agent, seeds the prompt, and
dispatches the run — **durably onto the worker queue when `agui.execution=worker`,
otherwise inline**. It returns the conversation and run ids so you can read the
result afterward:

```rust
// later — the agent's reply is the final assistant message
let messages = messages::Entity::find()
    .filter(messages::Column::ConversationId.eq(run.conversation_id))
    .all(&ctx.db).await?;
```

Two idiomatic ways to consume the result:

- **Read the reply** from `messages` once the run finishes (poll, or subscribe
  to the run's stream with the returned `run_id`).
- **Produce an artifact** — have the agent call the built-in `create_artifact`
  tool (see [Artifacts](#tool-context-artifacts-tenancy--tokens) below). The
  deliverable persists on the `artifacts` table and is fetchable at
  `GET conversations/{pid}/artifacts`, so you never read messages back. This is
  usually the better fit for report generation.

Pair `start_run` with the [scheduler](@/docs/processing/scheduler.md) for
recurring work (a nightly digest, a weekly report). For durability the same
requirements apply as any worker run: `workers.mode: BackgroundQueue` and a
DB-backed `agui.hub`.

> **Note:** the trigger message is persisted by the shared executor when the run
> actually starts — so with `execution=worker` the user message appears once a
> worker picks the job up, not at enqueue time. The HTTP `POST /run` endpoint
> and `start_run` behave identically here.

## Tools, hooks & authorization

- **Tools** (`src/agents/<name>/tools.rs`) are typed: each declares a `ToolSpec`
  once and receives deserialized arguments in `call`. Collect them into a
  `Tools` registry — specs and dispatch are derived, so there is no stringly
  typed `match` to maintain. A tool's `ToolKind` (`Read` / `Write`) drives
  approval gating.
- **Hooks** (`hooks.rs`) are observation / side-effect points around the run,
  each turn, and each tool call — metrics, auditing, redaction, cost tracking.
  A hook returning `Err` aborts the run.
- **Authorization** runs *before* hooks: an agent's `authorizer` can deny a tool
  call or require human approval. Approval surfaces as an AG-UI interrupt; the
  client answers by POSTing a `resume` instruction to `/run`.

## Tool context, artifacts, tenancy & tokens

Every tool call receives a `ToolContext` carrying the run's dependencies. It is
derived from the `AgentCtx` and rebuilt on the **executing node** (from the app
context, the principal, and the persisted conversation row), so nothing crosses
a serialized worker payload and inline/worker runs behave identically.

```rust
async fn call(&self, ctx: &ToolContext, args: MyArgs) -> Result<Value> {
    let app   = ctx.app();          // Option<&AppContext> — DB, storage, config
    let who   = &ctx.principal;     // the authenticated caller
    let scope = ctx.scope.as_ref(); // tenancy value (org/project), for billing/scoping
    let token = ctx.tokens();       // Option<Arc<dyn TokenResolver>>
    let deps  = ctx.ext::<MyDeps>(); // your custom deps (see below)
    // ...
}
```

- **Custom deps.** Override `Agent::extensions` to return an `Arc<dyn Any + Send +
  Sync>` (your own deps struct); recover it in a tool with `ctx.ext::<MyDeps>()`.
  It is built on the executing node, so it works under worker execution.
- **Token resolver.** Override `Agent::token_resolver` to return an
  `Arc<dyn TokenResolver>`; a tool calls `ctx.tokens().resolve("audience")` to
  mint/exchange a **fresh** access token — correct for long-running and durable
  runs whose captured principal would otherwise go stale.
- **Artifacts.** The framework injects built-in artifact tools —
  `create_artifact`, `update_artifact`, `get_artifact`, `list_artifacts`,
  `tag_artifact`. They persist to the `artifacts` table (id, name, kind,
  content/reference, `tags`, metadata, version), emit a `CUSTOM` event through the
  run hub (so a reconnecting client on any node sees them), and are fetchable at
  `GET conversations/{pid}/artifacts` and `.../artifacts/{artifact_pid}`.
- **Context & uploads.** Attach context to a conversation with
  `POST conversations/{pid}/context`, or upload a file with
  `POST conversations/{pid}/context/upload` (multipart → shared `Storage`). The
  agent reads them with the built-in `list_context` / `read_context` tools.

### Multitenancy (scoping conversations)

Conversations carry a `scope` JSON column. Tenant them by implementing a
`ScopeResolver` (compute the scope from request headers — e.g. an
`X-Organization-Id` — after loading/authorizing the org/project) and mounting it
with `routes_with_scope(registry, scope)`. The resolver stamps the scope on
create and filters every conversation read through one choke point, so a request
cannot reach a conversation outside its scope. The scope is read back off the row
on the executing node and exposed to tools via `ctx.scope`. Apps that own their
controller can call `service::create_conversation` / `service::find_conversation`
directly.

## Building blocks for richer agents

The framework injects a set of built-in tools into every run and exposes `Agent`
factories for the cross-cutting concerns a production, data-collating agent
needs:

- **Long-term memory (RAG).** `remember` / `search_memory` tools over a
  tenant-scoped `memories` table; `Agent::embedder` enables semantic (cosine)
  retrieval, else search is lexical. `cite` streams provenance.
- **Shared state.** `get_state` / `set_state` / `patch_state` persist structured
  working memory on the conversation and stream `STATE_SNAPSHOT` / `STATE_DELTA`
  so the UI reflects "the report so far" live.
- **Guardrails & budgets.** `Agent::guardrail` can rewrite or block model input
  and output (PII/injection/moderation); `Agent::budget` caps spend per
  tenant/run.
- **Deterministic workflows.** `SequentialAgent` / `ParallelAgent` / `LoopAgent`
  orchestrate child agents (pipe / fan-out / iterate) without an LLM in the loop.
- **Structured output.** `Agent::response_schema` constrains the answer to a JSON
  schema — reliable reports.
- **Planning & thinking.** `Agent::planner` (see `react_planner()`) adds a
  plan-then-act directive; the model's reasoning streams as `THINKING_CONTENT`.
- **Human-in-the-loop.** `ask_user` pauses for a clarifying answer;
  `suggest_followups` streams next-step prompts; long tools report progress via
  `ctx.progress()`.
- **Vision.** `multimodal_content(text, image_urls)` stores a user turn as
  image+text content for vision models.
- **Eval.** `EvalCase` + `run_suite` assert on the trajectory (tools called +
  output) for regression tests.

## What lives where

| Concern                              | Location                                   |
|--------------------------------------|--------------------------------------------|
| AG-UI wire events, message parts     | `loco_rs::agui::protocol`                  |
| LLM provider abstraction             | `loco_rs::agui::provider`                  |
| Run loop (`run_turn` / `resume`)     | `loco_rs::agui::runtime`                   |
| Typed tools                          | `loco_rs::agui::tool`                      |
| Tool context (deps/scope/tokens)     | `loco_rs::agui::context`                   |
| Built-in artifact tools              | `loco_rs::agui::artifact`                  |
| Built-in context/upload tools        | `loco_rs::agui::context_tool`              |
| Long-term memory (RAG)               | `loco_rs::agui::memory`                    |
| Shared-state tools                   | `loco_rs::agui::state_tool`                |
| Interaction tools (ask_user, ...)    | `loco_rs::agui::interact`                  |
| Guardrails + budgets                 | `loco_rs::agui::guardrail`                 |
| Workflow agents (seq/par/loop)       | `loco_rs::agui::workflow`                  |
| Trajectory eval harness              | `loco_rs::agui::eval`                      |
| Tenancy scope resolver               | `loco_rs::agui::scope`                     |
| Subagents                            | `loco_rs::agui::subagent`                  |
| Agent trait + registry + hooks       | `loco_rs::agui::agent`                     |
| Run hub (resumable/cancellable)      | `loco_rs::agui::hub`                        |
| Conversation store (DB)              | `loco_rs::agui::store`                      |
| Framework-owned entities             | `loco_rs::agui::entities`                   |
| Provider / prompt factories          | `loco_rs::agui::service`                    |
| HTTP router                          | `loco_rs::agui::controller`                 |
| Durable worker                       | `loco_rs::agui::worker`                     |
| **Your agents** (prompt/tools/hooks) | `src/agents/`                              |
