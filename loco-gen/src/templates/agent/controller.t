to: src/controllers/agents.rs
skip_exists: true
message: |
  Agent controller `agents` was added (a one-line mount of the framework router
  `loco_rs::agui::controller::routes`).

  Endpoints (all under `/api`):
    GET  agents                                  list declared agents
    GET  agents/{agent_id}                        one agent
    GET  agents/{agent_id}/conversations          list conversations
    POST agents/{agent_id}/conversations          open a conversation
    GET  conversations/{pid}/messages             message history
    POST conversations/{pid}/context              attach context
    POST conversations/{pid}/run                  start a run (SSE, resumable)
    GET  conversations/{pid}/stream?since=N        resume the live stream
    POST conversations/{pid}/cancel               stop the active run

  Frontend contract: POST `/run` to start and tail; on a dropped connection,
  GET `/stream?since=<lastEventId>` to resume (a fetch abort does NOT cancel the
  run); a Stop button calls POST `/cancel`.

  Durability: runs are driven inline by default. Set `agui.execution: { kind:
  worker }` (with `workers.mode: BackgroundQueue` and a DB-backed `agui.hub`) to
  hand each run to a background worker so it survives a process restart. Then
  register the worker in `connect_workers`:

      queue
          .register(loco_rs::agui::worker::RunAgentJob::with_registry(
              ctx,
              std::sync::Arc::new(crate::agents::registry()),
          ))
          .await?;
injections:
- into: src/controllers/mod.rs
  append: true
  content: "pub mod agents;"
- into: src/app.rs
  after: "AppRoutes::"
  content: "            .add_route(controllers::agents::routes())"
---
//! Thin HTTP mount for the AG-UI agent runtime. All agent logic — persistence,
//! provider, run hub, execution — lives in `loco_rs::agui`; this only hands the
//! framework router the registry of the app's declared agents.

use std::sync::Arc;

use loco_rs::prelude::*;

#[must_use]
pub fn routes() -> Routes {
    loco_rs::agui::controller::routes(Arc::new(crate::agents::registry()))
}
