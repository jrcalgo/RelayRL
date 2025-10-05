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

