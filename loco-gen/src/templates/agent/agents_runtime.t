to: src/agents/runtime.rs
skip_exists: true
---
//! Shared agent runtime wiring: the run-hub singleton (resumable/cancellable
//! streaming), config-driven provider construction, and system-prompt assembly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use loco_rs::agui::{
    channel_stream, in_memory, AgentCtx, AguiEvent, CancellationToken, HubEvent, HubEventStream,
    RigProvider, RunHandle, RunHub,
};
use loco_rs::config::HubConfig;
use loco_rs::prelude::*;
use sea_orm::QueryOrder;
use serde_json::Value;

use crate::models::_entities::{agent_events, agent_runs, context_items, conversations};

// ---------------------------------------------------------------------------
// Run hub singleton (chosen from `agui.hub` config)
// ---------------------------------------------------------------------------

static HUB: OnceLock<Arc<dyn RunHub>> = OnceLock::new();

/// The process-wide run hub. In-memory for single-node; DB-backed (multi-node)
/// when `agui.hub` is `redis` or `postgres`.
#[must_use]
pub fn run_hub(ctx: &AppContext) -> Arc<dyn RunHub> {
    HUB.get_or_init(|| {
        let kind = ctx
            .config
            .agui
            .as_ref()
            .map(|a| a.hub.clone())
            .unwrap_or_default();
        match kind {
            HubConfig::InMem => in_memory(),
            HubConfig::Redis | HubConfig::Postgres => Arc::new(DbRunHub::new(ctx.db.clone())),
        }
    })
    .clone()
}

/// Build the LLM provider from `agui.provider` config, defaulting the model to
/// the agent's declared model when config does not override it.
#[must_use]
pub fn provider(ctx: &AppContext, default_model: &str) -> RigProvider {
    match ctx.config.agui.as_ref() {
        Some(cfg) => {
            let model = cfg
                .provider
                .settings()
                .default_model
                .clone()
                .unwrap_or_else(|| default_model.to_string());
            RigProvider::from_config(&cfg.provider, model)
        }
        // No config: an empty-key provider (will fail on first call) — configure
        // `agui.provider` in config/*.yaml.
        None => RigProvider::new(String::new(), None, default_model.to_string()),
    }
}

/// Assemble a system prompt from a base string plus the conversation's mode and
/// any attached context items.
///
/// # Errors
/// Propagates DB errors while loading context items.
pub async fn assemble_system(ctx: &AgentCtx<'_>, base: &str) -> Result<String> {
    let mut parts = vec![base.to_string()];
    if let Ok(uuid) = Uuid::parse_str(&ctx.thread_id) {
        if let Some(conv) = conversations::Entity::find()
            .filter(conversations::Column::Pid.eq(uuid))
            .one(&ctx.app.db)
            .await?
        {
            if let Some(mode) = &ctx.mode {
                parts.push(format!("# Mode: {mode}"));
            }
            let items = context_items::Entity::find()
                .filter(context_items::Column::ConversationId.eq(conv.id))
                .all(&ctx.app.db)
                .await?;
            for item in items {
                if let Some(content) = item.content {
                    parts.push(format!("# Context: {}\n{content}", item.name));
                }
            }
        }
    }
    Ok(parts.join("\n\n"))
}

// ---------------------------------------------------------------------------
// DB-backed run hub (multi-node)
// ---------------------------------------------------------------------------

/// A multi-node [`RunHub`]: events persist to `agent_events` (replayed on
/// resume), and cancellation rides on `agent_runs.cancel_requested` — polled by
/// the node that owns the run, which flips its local token. Live tailing is by
/// polling the shared tables, so any node can serve a reconnect.
pub struct DbRunHub {
    db: DatabaseConnection,
    /// Per-run publish sequence (only the owning node publishes a given run).
    seqs: Mutex<HashMap<String, i64>>,
}

impl DbRunHub {
    #[must_use]
    pub fn new(db: DatabaseConnection) -> Self {
        Self {
            db,
            seqs: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl RunHub for DbRunHub {
    async fn start(&self, run_id: &str) -> Result<RunHandle> {
        let existing = agent_runs::Entity::find()
            .filter(agent_runs::Column::RunId.eq(run_id))
            .one(&self.db)
            .await?;
        if existing.is_none() {
            agent_runs::ActiveModel {
                pid: Set(Uuid::new_v4()),
                run_id: Set(run_id.to_string()),
                status: Set("running".to_string()),
                cancel_requested: Set(false),
                last_seq: Set(0),
                ..Default::default()
            }
            .insert(&self.db)
            .await?;
        }
        self.seqs
            .lock()
            .expect("seqs mutex")
            .insert(run_id.to_string(), 0);

        // Poll for a cross-node cancel request; flip the local token when seen.
        let cancel = CancellationToken::new();
        let poll_token = cancel.clone();
        let db = self.db.clone();
        let rid = run_id.to_string();
        tokio::spawn(async move {
            loop {
                if poll_token.is_cancelled() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1000)).await;
                let requested = agent_runs::Entity::find()
                    .filter(agent_runs::Column::RunId.eq(&rid))
                    .one(&db)
                    .await
                    .ok()
                    .flatten()
                    .is_some_and(|r| r.cancel_requested);
                if requested {
                    poll_token.cancel();
                    break;
                }
            }
        });

        Ok(RunHandle {
            run_id: run_id.to_string(),
            cancel,
        })
    }

    async fn publish(&self, run_id: &str, ev: &AguiEvent) -> Result<()> {
        let seq = {
            let mut m = self.seqs.lock().expect("seqs mutex");
            let e = m.entry(run_id.to_string()).or_insert(0);
            *e += 1;
            *e
        };
        let he = HubEvent::from_event(seq as u64, ev);
        agent_events::ActiveModel {
            pid: Set(Uuid::new_v4()),
            run_id: Set(run_id.to_string()),
            seq: Set(seq),
            name: Set(he.name.clone()),
            payload: Set(Some(he.data.clone())),
            ..Default::default()
        }
        .insert(&self.db)
        .await?;
        if let Some(row) = agent_runs::Entity::find()
            .filter(agent_runs::Column::RunId.eq(run_id))
            .one(&self.db)
            .await?
        {
            let mut am = row.into_active_model();
            am.last_seq = Set(seq);
            am.update(&self.db).await?;
        }
        Ok(())
    }

    async fn subscribe(&self, run_id: &str, since: u64) -> Result<HubEventStream> {
        let (tx, rx) = tokio::sync::mpsc::channel::<HubEvent>(256);
        let db = self.db.clone();
        let rid = run_id.to_string();
        tokio::spawn(async move {
            let mut last = i64::try_from(since).unwrap_or(0);
            loop {
                let events = agent_events::Entity::find()
                    .filter(agent_events::Column::RunId.eq(&rid))
                    .filter(agent_events::Column::Seq.gt(last))
                    .order_by_asc(agent_events::Column::Seq)
                    .all(&db)
                    .await
                    .unwrap_or_default();
                for e in events {
                    last = e.seq;
                    let he = HubEvent {
                        seq: e.seq as u64,
                        name: e.name,
                        data: e.payload.unwrap_or(Value::Null),
                    };
                    if tx.send(he).await.is_err() {
                        return; // client gone
                    }
                }
                let run_row = agent_runs::Entity::find()
                    .filter(agent_runs::Column::RunId.eq(&rid))
                    .one(&db)
                    .await
                    .ok()
                    .flatten();
                let done = match run_row {
                    Some(r) => matches!(r.status.as_str(), "complete" | "errored" | "cancelled"),
                    None => true,
                };
                if done {
                    // Final drain to catch events written just before terminal.
                    let tail = agent_events::Entity::find()
                        .filter(agent_events::Column::RunId.eq(&rid))
                        .filter(agent_events::Column::Seq.gt(last))
                        .order_by_asc(agent_events::Column::Seq)
                        .all(&db)
                        .await
                        .unwrap_or_default();
                    for e in tail {
                        let he = HubEvent {
                            seq: e.seq as u64,
                            name: e.name,
                            data: e.payload.unwrap_or(Value::Null),
                        };
                        let _ = tx.send(he).await;
                    }
                    return;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        });
        Ok(channel_stream(rx))
    }

    async fn cancel(&self, run_id: &str) -> Result<bool> {
        let Some(row) = agent_runs::Entity::find()
            .filter(agent_runs::Column::RunId.eq(run_id))
            .one(&self.db)
            .await?
        else {
            return Ok(false);
        };
        let mut am = row.into_active_model();
        am.cancel_requested = Set(true);
        am.status = Set("cancelling".to_string());
        am.update(&self.db).await?;
        Ok(true)
    }

    async fn finish(&self, run_id: &str) -> Result<()> {
        self.seqs.lock().expect("seqs mutex").remove(run_id);
        if let Some(row) = agent_runs::Entity::find()
            .filter(agent_runs::Column::RunId.eq(run_id))
            .one(&self.db)
            .await?
        {
            // Don't overwrite a cancelling/terminal status with "complete".
            if !matches!(row.status.as_str(), "cancelling" | "cancelled" | "errored") {
                let mut am = row.into_active_model();
                am.status = Set("complete".to_string());
                am.update(&self.db).await?;
            }
        }
        Ok(())
    }
}
