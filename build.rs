#[cfg(feature = "embedded_assets")]
use std::{env, path::Path};

fn main() {
    emit_build_info();

    #[cfg(feature = "embedded_assets")]
    embedded_assets_main();

    #[cfg(not(feature = "embedded_assets"))]
    {
        // No-op when feature is disabled
    }
}

/// Emit build-time metadata as compile-time environment variables.
///
/// These are baked into the binary and surfaced at runtime through the
/// `/_server` endpoint (see `src/server_info.rs`). Because Loco is compiled as
/// part of the host application's build, the profile, target, and `rustc`
/// version captured here match the application itself.
fn emit_build_info() {
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=LOCO_BUILD_PROFILE={profile}");

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=LOCO_BUILD_TARGET={target}");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = std::process::Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |v| v.trim().to_string());
    println!("cargo:rustc-env=LOCO_BUILD_RUSTC_VERSION={rustc_version}");
}

#[cfg(feature = "embedded_assets")]
fn embedded_assets_main() {
    // Import the embedded_assets module from the build directory
    #[path = "build/embedded_assets.rs"]
    mod embedded_assets;
    use embedded_assets::build_static_assets;

    // Get OUT_DIR environment variable - this is required for build scripts
    let out_dir = env::var("OUT_DIR").unwrap_or_else(|e| {
        // This should trigger a build failure
        panic!("OUT_DIR environment variable not set: {e}");
    });

    // Convert to a path
    let out_dir_path = Path::new(&out_dir);

    // Call the build_static_assets function with the OUT_DIR
    build_static_assets(out_dir_path);
}
