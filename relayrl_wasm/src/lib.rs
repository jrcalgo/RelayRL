/// **WASM Bindings for RL4Sys**: This module contains the Rust-to-WASM bindings,
/// exposing RL4Sys components as WASM-compatible classes. The `wasm_*` modules implement
/// wasm-bindgen-compatible wrappers for core structures, enabling smooth JavaScript interaction.
pub(crate) mod bindings {
    #[cfg(feature = "wasm_bindings")]
    pub(crate) mod wasm {
        pub(crate) mod wasm_action;
        pub(crate) mod wasm_configuration;
        pub(crate) mod wasm_trajectory;
        pub(crate) mod client {
            pub(crate) mod wasm_agent;
        }
    }
}