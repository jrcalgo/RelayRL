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
    // Builds the protocol buffer definitions
    #[cfg(feature = "grpc_network")]
    build_protobuf()?;

    Ok(())
}

/// Compile the protocol buffer definitions located in the specified proto file.
/// tonic_build::compile_protos will generate the corresponding Rust modules.
fn build_protobuf() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/relayrl.proto")?;
    Ok(())
}
