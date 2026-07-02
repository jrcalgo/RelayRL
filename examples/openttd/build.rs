use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=OPENTTD_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=OPENTTD_BUILD_DIR");

    if env::var_os("CARGO_FEATURE_NATIVE_OPENTTD").is_none() {
        return;
    }

    let source_dir = require_existing_dir(
        "OPENTTD_SOURCE_DIR",
        "OpenTTD 15.3 source checkout from https://github.com/OpenTTD/OpenTTD",
    );
    let build_dir = require_existing_dir(
        "OPENTTD_BUILD_DIR",
        "an OpenTTD 15.3 build directory containing libopenttd_relayrl_shim",
    );

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=dylib=openttd_relayrl_shim");
    println!("cargo:warning=Using OpenTTD source at {}", source_dir.display());
}

fn require_existing_dir(var: &str, description: &str) -> PathBuf {
    let value = env::var_os(var).unwrap_or_else(|| {
        panic!(
            "{var} must point to {description} when building with --features native-openttd"
        )
    });
    let path = PathBuf::from(value);
    if !path.is_dir() {
        panic!(
            "{var} must point to an existing directory for {description}; got {}",
            path.display()
        );
    }
    path
}
