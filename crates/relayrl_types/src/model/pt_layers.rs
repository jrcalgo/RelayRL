//! Read linear layers from a TorchScript module already loaded by `Model::load_from_file`.

use std::ffi::{CStr, c_char, c_void};
use std::mem::{align_of, size_of};

use tch::CModule;

use super::ModelError;

#[repr(C)]
struct RelayrlMlpLayer {
    in_dim: i64,
    out_dim: i64,
    weight_out_in: *mut f32,
    bias: *mut f32,
}

unsafe extern "C" {
    fn relayrl_pt_mlp_layers(
        module: *mut c_void,
        out_layers: *mut *mut RelayrlMlpLayer,
        out_count: *mut i32,
        out_error: *mut *mut c_char,
    ) -> i32;
    fn relayrl_pt_mlp_layers_free(layers: *mut RelayrlMlpLayer, count: i32);
    fn relayrl_pt_string_free(text: *mut c_char);
}

struct FreeLayers(*mut RelayrlMlpLayer, i32);

impl Drop for FreeLayers {
    fn drop(&mut self) {
        unsafe { relayrl_pt_mlp_layers_free(self.0, self.1) }
    }
}

/// Burn `[input, output]` layers from the loaded module's `aten::linear` nodes, in forward order.
pub(super) fn mlp_layers(
    module: &CModule,
) -> Result<Vec<(usize, usize, Vec<f32>, Vec<f32>)>, ModelError> {
    if size_of::<CModule>() != size_of::<*mut c_void>()
        || align_of::<CModule>() != align_of::<*mut c_void>()
    {
        return Err(ModelError::BackendError(
            "loaded TorchScript module pointer layout is unexpected".to_string(),
        ));
    }
    let module_ptr = unsafe { std::ptr::read(std::ptr::from_ref(module).cast::<*mut c_void>()) };
    let mut raw_layers = std::ptr::null_mut();
    let mut count = 0;
    let mut error = std::ptr::null_mut();
    let status = unsafe { relayrl_pt_mlp_layers(module_ptr, &mut raw_layers, &mut count, &mut error) };
    if status != 0 {
        let message = if error.is_null() {
            "failed to read linear layers from the loaded policy".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned();
            unsafe { relayrl_pt_string_free(error) };
            message
        };
        return Err(ModelError::BackendError(message));
    }
    let _guard = FreeLayers(raw_layers, count);
    if count < 0 || raw_layers.is_null() {
        return Err(ModelError::BackendError(
            "loaded policy returned no linear layers".to_string(),
        ));
    }
    let borrowed = unsafe { std::slice::from_raw_parts(raw_layers, count as usize) };
    let mut layers = Vec::with_capacity(borrowed.len());
    for layer in borrowed {
        let in_dim = usize::try_from(layer.in_dim).map_err(|_| {
            ModelError::InvalidInputDimension("policy input width does not fit usize".to_string())
        })?;
        let out_dim = usize::try_from(layer.out_dim).map_err(|_| {
            ModelError::InvalidOutputDimension("policy output width does not fit usize".to_string())
        })?;
        let weight_len = in_dim.checked_mul(out_dim).ok_or_else(|| {
            ModelError::InvalidInputDimension("policy weight length overflow".to_string())
        })?;
        if layer.weight_out_in.is_null() || layer.bias.is_null() {
            return Err(ModelError::BackendError(
                "loaded policy linear layer is missing weights".to_string(),
            ));
        }
        let weight = unsafe { std::slice::from_raw_parts(layer.weight_out_in, weight_len) };
        let bias = unsafe { std::slice::from_raw_parts(layer.bias, out_dim) };
        layers.push((
            in_dim,
            out_dim,
            pytorch_out_in_to_burn(weight, out_dim, in_dim),
            bias.to_vec(),
        ));
    }
    Ok(layers)
}

fn pytorch_out_in_to_burn(weight: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut burn = vec![0.0f32; in_dim * out_dim];
    for output in 0..out_dim {
        for input in 0..in_dim {
            burn[input * out_dim + output] = weight[output * in_dim + input];
        }
    }
    burn
}
