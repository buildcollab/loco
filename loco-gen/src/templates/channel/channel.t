{% set module_name = name | snake_case -%}
{% set struct_name = module_name | pascal_case -%}
to: "src/channels/{{module_name}}.rs"
skip_exists: true
message: "A channel `{{struct_name}}` was added. Mount it on a route, e.g. `Routes::new().add(\"/cable/{{module_name}}\", get(loco_rs::cable::transport::ws_handler::<crate::channels::{{module_name}}::{{struct_name}}>))`."
injections:
- into: "src/channels/mod.rs"
  append: true
  content: "pub mod {{ module_name }};"
- into: src/app.rs
  after: "fn register_channels"
  content: "        registry.register(\"{{module_name}}\", crate::channels::{{module_name}}::{{struct_name}}::default());"---
use loco_rs::prelude::*;

/// `{{struct_name}}` realtime channel.
///
/// Mount with:
/// ```ignore
/// Routes::new()
///     .add(
///         "/cable/{{module_name}}",
///         get(loco_rs::cable::transport::ws_handler::<crate::channels::{{module_name}}::{{struct_name}}>),
///     )
///     .add(
///         "/cable/{{module_name}}/sse",
///         get(loco_rs::cable::transport::sse_handler::<crate::channels::{{module_name}}::{{struct_name}}>),
///     )
/// ```
#[derive(Default)]
pub struct {{struct_name}};

#[async_trait]
impl Channel for {{struct_name}} {
    /// Connection-time params parsed from the query string. Use
    /// `serde_json::Value` if you don't want a typed shape.
    type Params = serde_json::Value;

    /// Return the topics this connection should stream from.
    async fn subscribed(&self, _ctx: &AppContext, _params: Self::Params) -> Result<Vec<String>> {
        Ok(vec!["{{module_name}}".to_string()])
    }

    /// Optional inbound (client → server) hook. Defaults to dropping data.
    async fn received(&self, _ctx: &AppContext, _data: bytes::Bytes) -> Result<()> {
        Ok(())
    }

    /// Optional cleanup on disconnect.
    async fn unsubscribed(&self, _ctx: &AppContext) -> Result<()> {
        Ok(())
    }
}
