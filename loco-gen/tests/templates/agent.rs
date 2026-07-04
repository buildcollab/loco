use super::utils::{guess_file_by_time, APP_ROUTS, MIGRATION_SRC_LIB};
use insta::{assert_snapshot, with_settings};
use loco_gen::{collect_messages, generate, tera_ext, AppInfo, Component};
use rrgen::RRgen;
use std::fs;

#[test]
fn can_generate() {
    let mut settings = insta::Settings::clone_current();
    settings.set_prepend_module_to_snapshot(false);
    settings.set_snapshot_suffix("agent");
    let _guard = settings.bind_to_scope();

    let component = Component::Agent {
        name: "support".to_string(),
        with_tz: true,
    };

    let tree_fs = tree_fs::TreeBuilder::default()
        .drop(true)
        .add_empty("src/controllers/mod.rs")
        .add("src/lib.rs", "pub mod controllers;\n")
        .add("migration/src/lib.rs", MIGRATION_SRC_LIB)
        .add("src/app.rs", APP_ROUTS)
        .create()
        .unwrap();

    let rrgen = RRgen::with_working_dir(&tree_fs.root).add_template_engine(tera_ext::new());

    let gen_result = generate(
        &rrgen,
        component,
        &AppInfo {
            app_name: "tester".to_string(),
        },
    )
    .expect("Generation failed");

    assert_snapshot!("generate_results", collect_messages(&gen_result));

    // MIGRATION
    let migration_path = tree_fs.root.join("migration/src");
    let migration_file = guess_file_by_time(&migration_path, "m{TIME}_agents.rs", 3)
        .expect("Failed to find the generated migration file");
    assert_snapshot!(
        "generate[migration_file]",
        fs::read_to_string(&migration_file).expect("Failed to read the migration file")
    );

    with_settings!({
        filters => vec![(r"\d{8}_\d{6}", "[TIME]")]
    }, {
        assert_snapshot!(
            "inject[migration_lib]",
            fs::read_to_string(migration_path.join("lib.rs")).expect("Failed to read lib.rs")
        );
    });

    // AGENTS MODULE (shared + per-agent files)
    let agents_path = tree_fs.root.join("src").join("agents");
    assert_snapshot!(
        "generate[agents_mod]",
        fs::read_to_string(agents_path.join("mod.rs")).expect("agents/mod.rs missing")
    );
    // `store.rs` and `runtime.rs` are no longer generated — that logic is now
    // library code in `loco_rs::agui` (store / hub / service / controller).
    assert_snapshot!(
        "generate[agent_dir_mod]",
        fs::read_to_string(agents_path.join("support").join("mod.rs")).expect("support/mod.rs missing")
    );
    assert_snapshot!(
        "generate[agent_dir_tools]",
        fs::read_to_string(agents_path.join("support").join("tools.rs"))
            .expect("support/tools.rs missing")
    );
    assert_snapshot!(
        "generate[agent_dir_hooks]",
        fs::read_to_string(agents_path.join("support").join("hooks.rs"))
            .expect("support/hooks.rs missing")
    );

    // CONTROLLER
    let controllers_path = tree_fs.root.join("src").join("controllers");
    assert_snapshot!(
        "generate[controller_file]",
        fs::read_to_string(controllers_path.join("agents.rs")).expect("controller file missing")
    );
    assert_snapshot!(
        "inject[controller_mod_rs]",
        fs::read_to_string(controllers_path.join("mod.rs")).expect("mod.rs injection failed")
    );

    // INJECTIONS into lib.rs (module decl + agent registration)
    assert_snapshot!(
        "inject[lib_rs]",
        fs::read_to_string(tree_fs.root.join("src").join("lib.rs")).expect("lib.rs injection failed")
    );
    assert_snapshot!(
        "inject[app_rs]",
        fs::read_to_string(tree_fs.root.join("src").join("app.rs")).expect("app.rs injection failed")
    );
}
