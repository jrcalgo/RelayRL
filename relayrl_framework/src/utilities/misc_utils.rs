use serde::{Deserialize, Serialize};
use std::any::TypeId;

macro_rules! arc {
    ($($elem:expr),*) => {{
        Arc::new([$($elem),*])
    }};
}

pub(crate) fn round_to_8_decimals<N>(num: N) -> N
where
    N: Copy + 'static,
{
    let size: usize = size_of::<N>();
    if TypeId::of::<N>() == TypeId::of::<f32>() && size == size_of::<f32>() {
        // Convert the generic N to f32.
        let n: f32 = unsafe { *(&num as *const N as *const f32) };
        let factor: f32 = 100_000_000.0_f32;
        let rounded: f32 = (n * factor).round() / factor;
        // Convert back to N.
        unsafe { *(&rounded as *const f32 as *const N) }
    } else if TypeId::of::<N>() == TypeId::of::<f64>() && size == size_of::<f64>() {
        // Convert the generic N to f64.
        let n: f64 = unsafe { *(&num as *const N as *const f64) };
        let factor: f64 = 100_000_000.0_f64;
        let rounded: f64 = (n * factor).round() / factor;
        // Convert back to N.
        unsafe { *(&rounded as *const f64 as *const N) }
    } else {
        panic!("Unsupported type. Only f32 and f64 are allowed.");
    }
}

/// A response received from the Python subprocess.
///
/// It contains a status string (e.g., "success") and an optional message.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct PythonResponse {
    pub(crate) status: String,
    message: Option<String>,
}
