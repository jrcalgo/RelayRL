/// **WASM Bindings for RelayRL**: This module contains the Rust-to-WASM bindings,
/// exposing RelayRL components as WASM-compatible classes. The `wasm_*` modules implement
/// wasm-bindgen-compatible wrappers for core structures, enabling smooth JavaScript interaction.
pub(crate) mod bindings {
    pub(crate) mod wasm {
        pub(crate) mod network {
            pub(crate) mod client {
                pub(crate) mod agent;
            }
        }
        pub(crate) mod types {
            pub(crate) mod action;
            pub(crate) mod trajectory;
        }
        pub(crate) mod utilities {
            pub(crate) mod configuration;
        }
    }
}
