to: src/agents/mod.rs
skip_exists: true
injections:
- into: src/lib.rs
  after: "pub mod controllers;"
  content: "pub mod agents;"
---
//! Agents — every code-declared agent lives under this module.
//!
//! Each agent is its own submodule implementing [`loco_rs::agui::Agent`]; the
//! declared name is the agent id stored on a conversation
//! (`conversations.agent_id`). Add one with `cargo loco generate agent <name>`.
//!
//! Persistence, the run hub, provider wiring, the HTTP router, and the durable
//! worker are all library code in `loco_rs::agui` — this module only declares
//! agents and builds their [`registry`].

// Per-agent modules (added by the generator):
// agents-mod-inject (do not remove)

use loco_rs::agui::AgentRegistry;

/// Build the registry of every declared agent. The controller resolves an
/// agent by `conversations.agent_id` against this registry.
#[must_use]
pub fn registry() -> AgentRegistry {
    #[allow(unused_mut)]
    let mut registry = AgentRegistry::new();
    // agents-register-inject (do not remove)
    registry
}
