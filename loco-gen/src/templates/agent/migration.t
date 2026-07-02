{% set mig_ts = ts | date(format="%Y%m%d_%H%M%S") -%}
to: "migration/src/m{{mig_ts}}_agents.rs"
skip_glob: "migration/src/m????????_??????_agents.rs"
message: "Agent migration added! Apply it with `$ cargo loco db migrate && cargo loco db entities`."
injections:
- into: "migration/src/lib.rs"
  before: "inject-above"
  content: "            Box::new(m{{mig_ts}}_agents::Migration),"
- into: "migration/src/lib.rs"
  before: "pub struct Migrator"
  content: "mod m{{mig_ts}}_agents;"
---
use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        // Agents: the configured assistants a user can talk to.
        create_table(
            m,
            "agents",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("name", ColType::String),
                ("description", ColType::TextNull),
                ("system_prompt", ColType::TextNull),
                ("provider", ColType::String),
                ("model", ColType::String),
                ("default_mode", ColType::StringNull),
                ("config", ColType::JsonBinaryNull),
            ],
            &[],
        )
        .await?;

        // User-defined modes: named overlays (extra system prompt / config).
        create_table(
            m,
            "agent_modes",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("name", ColType::String),
                ("system_prompt", ColType::TextNull),
                ("config", ColType::JsonBinaryNull),
            ],
            &[("agents", "")],
        )
        .await?;

        // Conversations: a chat thread with an agent.
        create_table(
            m,
            "conversations",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("title", ColType::StringNull),
                ("mode", ColType::StringNull),
                ("status", ColType::StringNull),
            ],
            &[("agents", "")],
        )
        .await?;

        // Messages: persisted turns (user / assistant / tool), with a `parts`
        // JSON blob capturing the canonical AG-UI message parts.
        create_table(
            m,
            "messages",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("role", ColType::String),
                ("content", ColType::TextNull),
                ("parts", ColType::JsonBinaryNull),
                ("provider", ColType::StringNull),
                ("model", ColType::StringNull),
                ("usage", ColType::JsonBinaryNull),
                ("status", ColType::StringNull),
            ],
            &[("conversations", "")],
        )
        .await?;

        // Tool calls: one row per tool invocation on an assistant message.
        create_table(
            m,
            "tool_calls",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("tool_call_id", ColType::String),
                ("name", ColType::String),
                ("arguments", ColType::JsonBinaryNull),
                ("status", ColType::String),
                ("result", ColType::JsonBinaryNull),
                ("duration_ms", ColType::BigIntegerNull),
            ],
            &[("messages", "")],
        )
        .await?;

        // Context items: files / system objects / text attached to a
        // conversation and folded into the system prompt.
        create_table(
            m,
            "context_items",
            &[
                ("id", ColType::PkAuto),
                ("pid", ColType::UuidUniq),
                ("kind", ColType::String),
                ("name", ColType::String),
                ("reference", ColType::StringNull),
                ("content", ColType::TextNull),
                ("metadata", ColType::JsonBinaryNull),
            ],
            &[("conversations", "")],
        )
        .await?;

        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        drop_table(m, "context_items").await?;
        drop_table(m, "tool_calls").await?;
        drop_table(m, "messages").await?;
        drop_table(m, "conversations").await?;
        drop_table(m, "agent_modes").await?;
        drop_table(m, "agents").await?;
        Ok(())
    }
}
