//! # Tool-invocation context, token resolution, and artifacts
//!
//! The run-loop's tool seam ([`ToolExecutor`](crate::agui::runtime::ToolExecutor)
//! / [`Tool`](crate::agui::tool::Tool)) is deliberately generic, but real tools
//! need request-scoped dependencies: the app context (DB, storage, config), the
//! authenticated [`Principal`], the tenancy `scope`, a place to emit events, a
//! way to persist [`Artifact`]s, fresh access tokens for external services, and
//! any app-defined custom deps. [`ToolContext`] carries all of that and is
//! handed to every tool call.
//!
//! ## Derived from [`AgentCtx`], rebuilt on the executing node
//!
//! A [`ToolContext`] is derived from the request-scoped [`AgentCtx`] via
//! [`AgentCtx::tool_context`], then augmented with the run's token resolver,
//! event sink, and artifact store. It is assembled in
//! [`worker::execute`](crate::agui::worker::execute) — which runs on the *executing*
//! node, so none of these (non-serializable) dependencies ride a durable job
//! payload; they are reconstructed from [`AppContext`] + the serialized
//! [`Principal`] + the persisted conversation row. This is what makes the whole
//! design correct under multi-node inline **and** worker execution.
//!
//! ## `Clone + Send + Sync + 'static`
//!
//! Each tool runs on its own [`tokio::spawn`] task, so the context must be
//! cloneable and `'static`; every field is owned or `Arc`-shared. The heavy
//! dependencies are `Option`al so a detached context (subagent / test runs,
//! [`ToolContext::default`]) needs no [`AppContext`] — tools that require one
//! surface a clean error instead.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::agui::agent::{AgentCtx, Principal};
use crate::agui::transport::EventSink;
use crate::app::AppContext;
use crate::{Error, Result};

/// Resolves access tokens for external services a tool needs to call.
///
/// Injected via [`ToolContext::tokens`] and supplied per-agent by
/// [`Agent::token_resolver`](crate::agui::agent::Agent::token_resolver). It is
/// **never serialized** — it is built on the executing node at run time, so a
/// long-running or worker-driven run mints/exchanges *fresh* tokens rather than
/// replaying a captured (and by then expired) one.
#[async_trait]
pub trait TokenResolver: Send + Sync {
    /// Resolve a token for `audience` (an app-defined key: an API name, an OAuth
    /// audience, a downstream service id, ...).
    ///
    /// # Errors
    /// Fails when the token cannot be minted/exchanged (network, denied, unknown
    /// audience).
    async fn resolve(&self, audience: &str) -> Result<String>;
}

/// A [`TokenResolver`] that resolves nothing — the default when an agent
/// declares none. Every `resolve` fails so a tool that actually needs a token
/// gets a clear error rather than a silent empty string.
pub struct NoTokens;

#[async_trait]
impl TokenResolver for NoTokens {
    async fn resolve(&self, audience: &str) -> Result<String> {
        Err(Error::Message(format!(
            "no token resolver configured (requested audience '{audience}')"
        )))
    }
}

/// A persisted artifact the agent produced — a document, report, or other
/// deliverable that outlives the message stream. Storage-agnostic: the payload
/// is either inline `content` or an external `reference` (e.g. a storage key).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    /// Public id (stable, shareable).
    pub pid: String,
    /// Human/model-facing name.
    pub name: String,
    /// Content type / kind (e.g. `text/markdown`, `report`), if any.
    pub kind: Option<String>,
    /// Inline textual content, if stored inline.
    pub content: Option<String>,
    /// Pointer to external content (e.g. a storage key), if not inline.
    pub reference: Option<String>,
    /// Free-form tags for organizing/fetching (e.g. `["draft"]`, `["published"]`).
    pub tags: Vec<String>,
    /// App-defined metadata.
    pub metadata: Option<Value>,
    /// Monotonic version, bumped on each update.
    pub version: i32,
}

/// Fields for creating a new [`Artifact`].
#[derive(Debug, Clone, Default)]
pub struct NewArtifact {
    pub name: String,
    pub kind: Option<String>,
    pub content: Option<String>,
    pub reference: Option<String>,
    pub tags: Vec<String>,
    pub metadata: Option<Value>,
}

/// Turns text into embedding vectors for semantic memory search. Supplied
/// per-agent by [`Agent::embedder`](crate::agui::agent::Agent::embedder) and used
/// by a [`MemoryStore`]. The default [`NoEmbedder`] returns no vectors, so the
/// store falls back to lexical ranking.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed each input string; returns one vector per input (or an empty `Vec`
    /// to signal "no embeddings — use lexical fallback").
    ///
    /// # Errors
    /// Fails when the embedding backend is unreachable.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// An [`Embedder`] that produces no vectors — the default. Memory search then
/// ranks by lexical token overlap instead of cosine similarity.
pub struct NoEmbedder;

#[async_trait]
impl Embedder for NoEmbedder {
    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }
}

/// A new memory to persist (a fact, a summary, a retrieved document chunk).
#[derive(Debug, Clone, Default)]
pub struct NewMemory {
    pub content: String,
    pub kind: Option<String>,
    pub metadata: Option<Value>,
}

/// A memory search result, ranked by relevance to the query.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryHit {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub kind: Option<String>,
    pub metadata: Option<Value>,
}

/// Long-term, retrievable memory for an agent — the RAG surface. Scoped by
/// tenant/conversation at construction; the framework provides a DB-backed
/// implementation
/// ([`DbMemoryStore`](crate::agui::store::DbMemoryStore)) that embeds on write
/// (when an [`Embedder`] is configured) and ranks on read by cosine similarity,
/// falling back to lexical overlap. Reached by memory
/// [`Tool`](crate::agui::tool::Tool)s through [`ToolContext::memory`].
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Persist memories (embedding them if an embedder is configured). Returns
    /// how many were stored.
    async fn add(&self, items: Vec<NewMemory>) -> Result<usize>;

    /// Retrieve the `top_k` memories most relevant to `query`.
    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<MemoryHit>>;
}

/// Persistence for a conversation's [`Artifact`]s. Reached by artifact
/// [`Tool`](crate::agui::tool::Tool)s through [`ToolContext::artifacts`]; the
/// framework provides a DB-backed implementation
/// ([`DbArtifactStore`](crate::agui::store::DbArtifactStore)). Kept separate
/// from [`ConversationStore`](crate::agui::runtime::ConversationStore) so the
/// run-loop core stays free of the artifact concept.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    /// Create a new artifact (version 1) for the current conversation.
    async fn create(&self, new: NewArtifact) -> Result<Artifact>;

    /// Update an artifact's content/tags/metadata by `pid`, bumping its version.
    /// `None` fields are left unchanged.
    async fn update(
        &self,
        pid: &str,
        content: Option<String>,
        tags: Option<Vec<String>>,
        metadata: Option<Value>,
    ) -> Result<Artifact>;

    /// Fetch one artifact by `pid` (scoped to this conversation).
    async fn get(&self, pid: &str) -> Result<Option<Artifact>>;

    /// List the conversation's artifacts, optionally filtered to those carrying
    /// `tag`.
    async fn list(&self, tag: Option<&str>) -> Result<Vec<Artifact>>;
}

/// Request-scoped context handed to every tool call.
///
/// Derive one from an [`AgentCtx`] with [`AgentCtx::tool_context`], then attach
/// the run's [`TokenResolver`], [`EventSink`], and [`ArtifactStore`] with the
/// `with_*` builders. The heavy dependencies are `Option`al so a detached
/// context ([`ToolContext::default`], used by subagent/test runs that have no
/// [`AppContext`]) still constructs; the accessors return `None` and tools that
/// need a dependency error cleanly.
#[derive(Clone, Default)]
pub struct ToolContext {
    app: Option<AppContext>,
    /// The authenticated caller.
    pub principal: Principal,
    /// Public id of the conversation (AG-UI `thread_id`).
    pub thread_id: String,
    /// Numeric id of the conversation (for store/artifact scoping).
    pub conversation_id: i32,
    /// The run id (AG-UI `run_id`).
    pub run_id: String,
    /// The persisted tenancy value (org/project/...), read from the conversation
    /// row. Present for scoping and billing inside tools.
    pub scope: Option<Value>,
    tokens: Option<Arc<dyn TokenResolver>>,
    sink: Option<Arc<dyn EventSink>>,
    artifacts: Option<Arc<dyn ArtifactStore>>,
    memory: Option<Arc<dyn MemoryStore>>,
    extensions: Option<Arc<dyn Any + Send + Sync>>,
}

impl ToolContext {
    /// The app context (DB, storage, config), if this is a live run context.
    #[must_use]
    pub fn app(&self) -> Option<&AppContext> {
        self.app.as_ref()
    }

    /// The run's token resolver, if configured.
    #[must_use]
    pub fn tokens(&self) -> Option<Arc<dyn TokenResolver>> {
        self.tokens.clone()
    }

    /// The run's event sink (for emitting protocol events from a tool), if any.
    #[must_use]
    pub fn sink(&self) -> Option<Arc<dyn EventSink>> {
        self.sink.clone()
    }

    /// The conversation's artifact store, if configured.
    #[must_use]
    pub fn artifacts(&self) -> Option<Arc<dyn ArtifactStore>> {
        self.artifacts.clone()
    }

    /// The agent's long-term memory store (RAG), if configured.
    #[must_use]
    pub fn memory(&self) -> Option<Arc<dyn MemoryStore>> {
        self.memory.clone()
    }

    /// Downcast the app-supplied custom deps to `T` (the type an
    /// [`Agent::extensions`](crate::agui::agent::Agent::extensions) built).
    #[must_use]
    pub fn ext<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.extensions.as_ref().and_then(|e| e.downcast_ref::<T>())
    }

    /// Attach the run's token resolver. Builder style.
    #[must_use]
    pub fn with_tokens(mut self, tokens: Arc<dyn TokenResolver>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Attach the run's event sink. Builder style.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Attach the conversation's artifact store. Builder style.
    #[must_use]
    pub fn with_artifacts(mut self, artifacts: Arc<dyn ArtifactStore>) -> Self {
        self.artifacts = Some(artifacts);
        self
    }

    /// Attach the agent's memory store. Builder style.
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn MemoryStore>) -> Self {
        self.memory = Some(memory);
        self
    }
}

impl AgentCtx<'_> {
    /// Derive a [`ToolContext`] for a run from this request-scoped context,
    /// carrying the app context, principal, thread/conversation ids, scope, and
    /// the app's custom `extensions`. Attach the run's token resolver / sink /
    /// artifact store afterward with the `with_*` builders.
    #[must_use]
    pub fn tool_context(&self, run_id: impl Into<String>) -> ToolContext {
        ToolContext {
            app: Some(self.app.clone()),
            principal: self.principal.clone(),
            thread_id: self.thread_id.clone(),
            conversation_id: self.conversation_id,
            run_id: run_id.into(),
            scope: self.scope.clone(),
            extensions: Some(self.extensions.clone()),
            tokens: None,
            sink: None,
            artifacts: None,
            memory: None,
        }
    }
}

#[cfg(all(test, feature = "agui"))]
mod tests {
    use super::*;

    struct Deps {
        api_key: String,
    }

    #[test]
    fn ext_downcast_and_detached_defaults() {
        let ctx = ToolContext {
            extensions: Some(Arc::new(Deps {
                api_key: "secret".to_string(),
            })),
            ..Default::default()
        };
        assert_eq!(ctx.ext::<Deps>().unwrap().api_key, "secret");
        // Wrong type downcasts to None.
        assert!(ctx.ext::<String>().is_none());

        // A default (detached) context has no heavy deps.
        let detached = ToolContext::default();
        assert!(detached.app().is_none());
        assert!(detached.tokens().is_none());
        assert!(detached.artifacts().is_none());
        assert!(detached.sink().is_none());
        assert!(detached.ext::<Deps>().is_none());
    }

    #[tokio::test]
    async fn no_tokens_resolver_errors() {
        let err = NoTokens.resolve("some-api").await.unwrap_err();
        assert!(err.to_string().contains("no token resolver configured"));
    }
}
