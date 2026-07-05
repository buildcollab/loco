//! # Framework-owned agent entities
//!
//! SeaORM entities for the agent-runtime tables (`conversations`, `messages`,
//! `tool_calls`, `context_items`, `agent_runs`, `agent_events`). Keeping these
//! in the framework is what lets [`DbStore`](super::store::DbStore) and
//! [`DbRunHub`](super::hub::DbRunHub) be **library** code rather than generated
//! into every app: the app owns the schema (the generated migration creates the
//! tables) but the mapping to the run-loop lives here.
//!
//! The shapes mirror the generated `agent` migration exactly. Rows carry the
//! framework-managed `created_at` / `updated_at` timestamp columns in the
//! database; they are intentionally omitted here because the run-loop never
//! reads them and the database supplies their defaults on insert.

#![allow(clippy::doc_markdown)]

pub mod conversations {
    //! A chat thread bound to a code-declared agent.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "conversations")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        /// The agent's registry name (see `conversations.agent_id`).
        pub agent_id: String,
        pub title: Option<String>,
        pub mode: Option<String>,
        pub status: Option<String>,
        /// The in-flight run id, so a client can resume or cancel it.
        pub active_run_id: Option<String>,
        /// App-defined tenancy value (e.g. `{organization_id, project_id}`), set
        /// by the controller at create and used to scope reads. `NULL` = unscoped.
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub scope: Option<Json>,
        /// Free-form app extensibility slot for non-scoping data.
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub metadata: Option<Json>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod messages {
    //! A persisted turn (user / assistant / tool).
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "messages")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        pub conversation_id: i32,
        pub role: String,
        pub content: Option<String>,
        /// Canonical AG-UI message parts; round-trips into provider history.
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub parts: Option<Json>,
        pub provider: Option<String>,
        pub model: Option<String>,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub usage: Option<Json>,
        pub status: Option<String>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod tool_calls {
    //! One row per tool invocation on an assistant message.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "tool_calls")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        pub message_id: i32,
        pub tool_call_id: String,
        pub name: String,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub arguments: Option<Json>,
        pub status: String,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub result: Option<Json>,
        pub duration_ms: Option<i64>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod context_items {
    //! Files / system objects / text attached to a conversation and folded into
    //! the system prompt.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "context_items")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        pub conversation_id: i32,
        /// Optional message this context item is attached to (message-scoped
        /// context/attachments). `NULL` = conversation-scoped.
        pub message_id: Option<i32>,
        pub kind: String,
        pub name: String,
        pub reference: Option<String>,
        pub content: Option<String>,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub metadata: Option<Json>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod agent_runs {
    //! Run registry for the multi-node run hub: one row per run.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "agent_runs")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        pub run_id: String,
        pub conversation_id: Option<i32>,
        pub status: String,
        /// Cross-node cancel flag; the owning node polls it and flips its token.
        pub cancel_requested: bool,
        /// Newest published event sequence number.
        pub last_seq: i64,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod agent_events {
    //! The ordered per-run event log the run hub replays on reconnect.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "agent_events")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        pub run_id: String,
        pub seq: i64,
        pub name: String,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub payload: Option<Json>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod memories {
    //! Long-term, retrievable memory (RAG). Scoped by tenant (`scope`) and/or a
    //! conversation; `embedding` holds a JSON array of floats when an embedder is
    //! configured, else search falls back to lexical ranking.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "memories")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        /// Tenant scope (matches `conversations.scope`); `NULL` = global.
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub scope: Option<Json>,
        /// Optional owning conversation; `NULL` = tenant/global memory.
        pub conversation_id: Option<i32>,
        pub kind: Option<String>,
        #[sea_orm(column_type = "Text")]
        pub content: String,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub embedding: Option<Json>,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub metadata: Option<Json>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod artifacts {
    //! A persistent deliverable (document / report) the agent produced, scoped to
    //! a conversation. Fetchable for display and mutable by the built-in artifact
    //! tools.
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, serde::Serialize)]
    #[sea_orm(table_name = "artifacts")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pid: Uuid,
        pub conversation_id: i32,
        pub name: String,
        pub kind: Option<String>,
        pub content: Option<String>,
        pub reference: Option<String>,
        /// Free-form tags (JSON array of strings) for organizing/fetching.
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub tags: Option<Json>,
        #[sea_orm(column_type = "JsonBinary", nullable)]
        pub metadata: Option<Json>,
        pub version: i32,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}
