//! # Conversation tenancy scope
//!
//! Multitenancy for conversations is a *request-set, persisted* value — not
//! something derived from the [`Principal`] alone, because the tenant is
//! frequently request data (an `X-Organization-Id` / `X-Project-Id` header) that
//! the controller loads and authorizes. A [`ScopeResolver`] computes that value
//! for a request; the controller stamps it on the `conversations.scope` column
//! at create and filters reads by it. The value is then re-read from the row on
//! the executing node and threaded into the
//! [`ToolContext`](crate::agui::context::ToolContext) for scoping/billing —
//! correct under multi-node inline and worker execution.
//!
//! The default [`NoScope`] leaves conversations unscoped (today's behavior). An
//! app implements [`ScopeResolver`] to tenant on an organization/project/user,
//! and either mounts it on the built-in controller
//! ([`routes_with_scope`](crate::agui::controller::routes_with_scope)) or calls
//! the [`service`](crate::agui::service) helpers from its own controller.

use async_trait::async_trait;
use sea_orm::sea_query::{extension::postgres::PgBinOper, Expr};
use sea_orm::{ColumnTrait, Condition, ExprTrait};
use serde_json::Value;

use super::agent::Principal;
use super::entities::conversations;
use crate::app::AppContext;
use crate::Result;

/// A Postgres JSONB-containment (`@>`) condition: select rows whose `column`
/// *contains* every key/value in `subset`.
///
/// The default [`ScopeResolver::filter`] matches the scope column *exactly*,
/// which is the right default (portable, and an exact tenant match is the
/// common case). It is too strict, though, when a conversation is stamped with
/// a rich scope (e.g. `{organization_id, project_id}`) but must also be visible
/// to a coarser query — an org-wide history rail that filters on just
/// `{organization_id}`. Override `filter` with this helper to get that:
///
/// ```ignore
/// use loco_rs::agui::scope;
/// # use loco_rs::agui::entities::conversations;
/// # use sea_orm::Condition; use serde_json::Value;
/// fn filter(&self, scope: &Value) -> Condition {
///     scope::contains(conversations::Column::Scope, scope)
/// }
/// ```
///
/// Postgres-only: `@>` is not portable to SQLite/MySQL. Pair it with the GIN
/// index the generated agent migration builds on the `scope` columns so the
/// containment probe stays index-backed instead of seq-scanning at scale.
#[must_use]
pub fn contains<C: ColumnTrait>(column: C, subset: &Value) -> Condition {
    Condition::all().add(Expr::col(column).binary(PgBinOper::Contains, Expr::val(subset.clone())))
}

/// Computes the tenancy scope for a request and the DB filter that restricts
/// which conversations a request may see.
#[async_trait]
pub trait ScopeResolver: Send + Sync {
    /// Resolve the scope value for the current request — e.g. read the
    /// `X-Organization-Id` / `X-Project-Id` headers, load + authorize the
    /// entities, and return `{"organization_id": .., "project_id": ..}`.
    /// `None` = unscoped (no filtering, no stamp).
    ///
    /// # Errors
    /// Return an error (surfaced as an HTTP error) to reject the request — e.g.
    /// a missing/invalid organization header, or an unauthorized project.
    async fn resolve(
        &self,
        ctx: &AppContext,
        parts: &axum::http::request::Parts,
        principal: &Principal,
    ) -> Result<Option<Value>>;

    /// The DB condition selecting conversations visible under `scope`. The
    /// default matches rows whose `scope` column equals `scope` exactly;
    /// override for JSON-containment (Postgres `scope @> ..`, via the
    /// [`contains`] helper) or for a relational scheme where the app added its
    /// own columns.
    fn filter(&self, scope: &Value) -> Condition {
        Condition::all().add(conversations::Column::Scope.eq(scope.clone()))
    }
}

/// The default [`ScopeResolver`]: conversations are unscoped, preserving the
/// pre-tenancy behavior (any caller who knows a conversation `pid` may access
/// it). Apps layer their own auth over the routes, or supply a real resolver.
pub struct NoScope;

#[async_trait]
impl ScopeResolver for NoScope {
    async fn resolve(
        &self,
        _ctx: &AppContext,
        _parts: &axum::http::request::Parts,
        _principal: &Principal,
    ) -> Result<Option<Value>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use sea_orm::{DatabaseBackend, EntityTrait, QueryFilter, QueryTrait};
    use serde_json::json;

    use super::{contains, conversations};

    #[test]
    fn contains_builds_jsonb_containment_sql() {
        let cond = contains(conversations::Column::Scope, &json!({"organization_id": 7}));
        let sql = conversations::Entity::find()
            .filter(cond)
            .build(DatabaseBackend::Postgres)
            .to_string();
        assert!(sql.contains("@>"), "expected JSONB containment, got: {sql}");
        assert!(
            sql.contains("scope"),
            "should filter the scope column: {sql}"
        );
    }
}
