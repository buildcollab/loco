//! # Built-in shared-state tools
//!
//! An agent's *shared state* is a structured, evolving JSON object (working
//! memory) persisted on the conversation and streamed to the UI as AG-UI
//! `STATE_SNAPSHOT` / `STATE_DELTA` events — the right home for "the report I'm
//! building" or "the filters the user chose", rather than burying it in the
//! transcript.
//!
//! `get_state` reads it, `set_state` replaces it (emits a snapshot), and
//! `patch_state` shallow-merges an object into it (emits a delta). Composed into
//! a run by [`worker::execute`](crate::agui::worker::execute) via
//! [`builtin_state_tools`].

use async_trait::async_trait;
use sea_orm::{ActiveModelTrait, EntityTrait, IntoActiveModel, Set};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::entities::conversations;
use crate::agui::context::ToolContext;
use crate::agui::protocol::AguiEvent;
use crate::agui::provider::{ToolKind, ToolSpec};
use crate::agui::tool::{NoArgs, Tool, Tools};
use crate::{Error, Result};

/// The framework's built-in shared-state tools: `get_state`, `set_state`,
/// `patch_state`.
#[must_use]
pub fn builtin_state_tools() -> Tools {
    Tools::new().with(GetState).with(SetState).with(PatchState)
}

async fn load_state(ctx: &ToolContext) -> Result<Value> {
    let db = ctx
        .app()
        .map(|a| &a.db)
        .ok_or_else(|| Error::string("state tools require app context"))?;
    let row = conversations::Entity::find_by_id(ctx.conversation_id)
        .one(db)
        .await?
        .ok_or(Error::NotFound)?;
    Ok(row.state.unwrap_or_else(|| json!({})))
}

async fn store_state(ctx: &ToolContext, state: Value) -> Result<()> {
    let db = ctx
        .app()
        .map(|a| &a.db)
        .ok_or_else(|| Error::string("state tools require app context"))?;
    let row = conversations::Entity::find_by_id(ctx.conversation_id)
        .one(db)
        .await?
        .ok_or(Error::NotFound)?;
    let mut am = row.into_active_model();
    am.state = Set(Some(state));
    am.update(db).await?;
    Ok(())
}

/// Emit the full state as a `STATE_SNAPSHOT`.
async fn emit_snapshot(ctx: &ToolContext, state: &Value) {
    if let Some(sink) = ctx.sink() {
        let _ = sink
            .emit(AguiEvent::StateSnapshot {
                snapshot: state.clone(),
            })
            .await;
    }
}

/// Emit an incremental `STATE_DELTA`.
async fn emit_delta(ctx: &ToolContext, delta: &Value) {
    if let Some(sink) = ctx.sink() {
        let _ = sink
            .emit(AguiEvent::StateDelta {
                delta: delta.clone(),
            })
            .await;
    }
}

/// Shallow merge `patch`'s keys into `base` (RFC 7386-style, one level).
fn shallow_merge(base: &mut Value, patch: &Value) {
    if let (Some(b), Some(p)) = (base.as_object_mut(), patch.as_object()) {
        for (k, v) in p {
            if v.is_null() {
                b.remove(k);
            } else {
                b.insert(k.clone(), v.clone());
            }
        }
    } else {
        *base = patch.clone();
    }
}

/// Read the conversation's shared state.
struct GetState;

#[async_trait]
impl Tool for GetState {
    type Args = NoArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_state".to_string(),
            description: "Read the conversation's shared state object (structured working memory)."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Read,
        }
    }

    async fn call(&self, ctx: &ToolContext, _args: NoArgs) -> Result<Value> {
        Ok(json!({ "state": load_state(ctx).await? }))
    }
}

#[derive(Deserialize)]
struct SetStateArgs {
    state: Value,
}

/// Replace the conversation's shared state entirely.
struct SetState;

#[async_trait]
impl Tool for SetState {
    type Args = SetStateArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "set_state".to_string(),
            description:
                "Replace the conversation's shared state object entirely. Streams a state \
                          snapshot to the UI."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "state": { "type": "object", "description": "The new state object." } },
                "required": ["state"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: SetStateArgs) -> Result<Value> {
        store_state(ctx, args.state.clone()).await?;
        emit_snapshot(ctx, &args.state).await;
        Ok(json!({ "ok": true, "state": args.state }))
    }
}

#[derive(Deserialize)]
struct PatchStateArgs {
    patch: Value,
}

/// Shallow-merge an object into the conversation's shared state.
struct PatchState;

#[async_trait]
impl Tool for PatchState {
    type Args = PatchStateArgs;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "patch_state".to_string(),
            description: "Merge keys into the conversation's shared state (a null value removes a \
                          key). Streams a state delta to the UI."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "patch": { "type": "object", "description": "Keys to merge into state." } },
                "required": ["patch"]
            }),
            kind: ToolKind::Write,
        }
    }

    async fn call(&self, ctx: &ToolContext, args: PatchStateArgs) -> Result<Value> {
        let mut state = load_state(ctx).await?;
        if !state.is_object() {
            state = Value::Object(Map::new());
        }
        shallow_merge(&mut state, &args.patch);
        store_state(ctx, state.clone()).await?;
        emit_delta(ctx, &args.patch).await;
        Ok(json!({ "ok": true, "state": state }))
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;

    #[test]
    fn shallow_merge_sets_and_removes() {
        let mut base = json!({ "a": 1, "b": 2 });
        shallow_merge(&mut base, &json!({ "b": 3, "c": 4, "a": null }));
        assert_eq!(base, json!({ "b": 3, "c": 4 }));
    }
}
