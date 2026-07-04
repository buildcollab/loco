{% set mod_name = name | snake_case -%}
{% set struct_name = name | pascal_case -%}
to: src/agents/{{mod_name}}/mod.rs
skip_exists: true
message: |
  Agent `{{mod_name}}` scaffolded under `src/agents/{{mod_name}}/`.

  Next steps:
    1. Enable the `agui` feature on loco-rs in Cargo.toml:
         loco-rs = { version = "*", features = ["agui"] }
    2. Apply the migration + entity sync:
         $ cargo loco db migrate && cargo loco db entities
    3. Configure the provider + hub under `agui:` in config/*.yaml (see the
       `agui.provider.api_key` / `agui.hub` keys).
    4. Open a conversation for agent id `{{mod_name}}` and POST to
         /api/conversations/{conversation_pid}/run
       with an AG-UI `RunAgentInput` body to stream a response.

  Customize this agent's prompt/model here, its tools in `tools.rs`, and its
  lifecycle callbacks in `hooks.rs`.
injections:
- into: src/agents/mod.rs
  after: "// agents-mod-inject (do not remove)"
  content: "pub mod {{mod_name}};"
- into: src/agents/mod.rs
  after: "// agents-register-inject (do not remove)"
  content: "    registry.register({{mod_name}}::{{struct_name}}Agent);"
---
//! The `{{mod_name}}` agent.

pub mod hooks;
pub mod tools;

use std::sync::Arc;

use async_trait::async_trait;
use loco_rs::agui::{Agent, AgentCtx, AgentHooks, Tools};
use loco_rs::prelude::*;

/// Base system prompt for this agent. Edit freely; conversation context items
/// are appended by [`crate::agents::runtime::assemble_system`].
const SYSTEM_PROMPT: &str = "You are {{mod_name}}, a helpful assistant.";

/// Default model. Overridden by `agui.provider.default_model` when set.
const MODEL: &str = "anthropic/claude-sonnet-5";

/// The `{{mod_name}}` agent definition.
pub struct {{struct_name}}Agent;

#[async_trait]
impl Agent for {{struct_name}}Agent {
    fn name(&self) -> &str {
        "{{mod_name}}"
    }

    fn description(&self) -> &str {
        "The {{mod_name}} agent."
    }

    fn model(&self) -> String {
        MODEL.to_string()
    }

    async fn system_prompt(&self, ctx: &AgentCtx<'_>) -> Result<String> {
        crate::agents::runtime::assemble_system(ctx, SYSTEM_PROMPT).await
    }

    fn tools(&self) -> Tools {
        tools::tools()
    }

    fn hooks(&self) -> Arc<dyn AgentHooks> {
        Arc::new(hooks::{{struct_name}}Hooks)
    }
}
