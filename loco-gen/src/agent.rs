use std::path::Path;

use chrono::Utc;
use rrgen::RRgen;
use serde_json::json;

use crate::{render_template, AppInfo, GenerateResults, Result};

/// Generate the AG-UI agent subsystem: a migration for the agent tables and a
/// self-contained controller wiring them to `loco_rs::agui`.
///
/// The generated code targets `loco_rs::agui` (enable the `agui` feature on
/// loco-rs) and depends on the entities produced by
/// `cargo loco db migrate && cargo loco db entities`.
///
/// # Errors
/// Returns an error if the templates cannot be rendered.
pub fn generate(
    rrgen: &RRgen,
    name: &str,
    with_tz: bool,
    appinfo: &AppInfo,
) -> Result<GenerateResults> {
    let ts = Utc::now();
    let vars = json!({
        "pkg_name": appinfo.app_name,
        "name": name,
        "ts": ts,
        "with_tz": with_tz,
    });
    render_template(rrgen, Path::new("agent"), &vars)
}
