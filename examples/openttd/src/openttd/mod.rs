mod mock;
mod types;

#[cfg(feature = "native-openttd")]
mod ffi;
#[cfg(feature = "native-openttd")]
mod native;

pub use types::*;

#[cfg(not(feature = "native-openttd"))]
pub use mock::OpenTtdBridge;

#[cfg(feature = "native-openttd")]
pub use native::OpenTtdBridge;
