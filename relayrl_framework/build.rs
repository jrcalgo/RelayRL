use std::fs::create_dir_all;
use std::path::Path;
use std::process::Command;

/// The main function used for building the project.
///
/// This function compiles the protocol buffer definitions specified in the
/// "proto/relayrl.proto" file using tonic_build. The generated
/// Rust code is then used for gRPC communication within the RelayRL framework.
///
/// # Returns
///
/// * `Ok(())` if the proto compilation succeeds.
/// * An error of type `Box<dyn std::error::Error>` if compilation fails.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    pyo3_build_config::add_extension_module_link_args();

    // Builds the protocol buffer definitions
    #[cfg(feature = "grpc_network")]
    build_protobuf()?;

    // TODO: Builds python bindings for PyOxidizer binary (this is w.i.p.)
    // without this, assume end-user has an installation of Python capable of using
    // maturin and PyTorch == 2.5.1. Different build instructions may be needed
    // #[cfg(feature = "compile_python_binary")]
    // {
    //     build_data_bindings()?;
    //     build_python_binary()?;
    // }

    Ok(())
}

/// Compile the protocol buffer definitions located in the specified proto file.
/// tonic_build::compile_protos will generate the corresponding Rust modules.
fn build_protobuf() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/relayrl.proto")?;
    Ok(())
}

/// This is specifically for building the `training_data` component bindings, which includes
/// [RelayRLAction] and [RelayRLTrajectory] implementations.
fn build_data_bindings() -> Result<(), Box<dyn std::error::Error>> {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Check if maturin is installed
    let maturin_check = Command::new("maturin").arg("--version").output();

    if maturin_check.is_err() || !maturin_check.unwrap().status.success() {
        panic!("Maturin not found. Please install it with `pip install maturin`");
    }

    // Build Python bindings in development mode so they're available in the Python environment
    let status = Command::new("maturin")
        .arg("develop")
        .arg("--release")
        .arg("--features=training_data")
        .current_dir(&crate_dir)
        .status()?;

    if !status.success() {
        panic!("Failed to build Python bindings, exit status: {}", status);
    }

    println!("Successfully built Python bindings");

    Ok(())
}

///
///
fn build_python_binary() -> Result<(), Box<dyn std::error::Error>> {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let python_dir = Path::new(&crate_dir).join("src/native/python");
    let binary_output_dir = python_dir.join("bin");

    create_dir_all(&binary_output_dir)?;

    // Check for existence/version
    let pyoxidizer_check = Command::new("pyoxidizer").arg("--version").output();
    if pyoxidizer_check.is_err() || !pyoxidizer_check.unwrap().status.success() {
        // As of now, must hard fail - Rust impl necessitates a python binary for end-user ease
        panic!("PyOxidizer not found. Please install it with `cargo install pyoxidizer`");
    }

    println!("cargo:rerun-if-changed=src/native/python");

    // Build binary
    let status = Command::new("pyoxidizer")
        .arg("build")
        .current_dir(&python_dir)
        .status()?;
    if !status.success() {
        // Hard fail - RelayRL Rust impl cannot function (right now) without a python binary
        panic!("Failed to build python binary, exit status: {}", status);
    }

    println!("Successfully built Python binary.");

    Ok(())
}
