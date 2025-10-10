relayrl_types
==============

Core data types for the RelayRL framework.

- RelayRLAction: serializable action container (obs, act, mask, reward, data, done)
- RelayRLTrajectory: in-memory trajectory buffer with a max length
- Safetensors-based tensor (de)serialization helpers

Install
-------

Add to your Cargo.toml:

```toml
relayrl_types = { path = "../relayrl_types" }  # or version
```

Quick start
-----------

```rust
use relayrl_types::types::action::RelayRLAction;
use relayrl_types::types::trajectory::{RelayRLTrajectory, RelayRLTrajectoryTrait};
use tch::{Device, Kind, Tensor};

// Create some tensors
let obs = Tensor::randn([4], (Kind::Float, Device::Cpu));
let act = Tensor::zeros([2], (Kind::Float, Device::Cpu));

// Serialize tensors into a RelayRLAction
let step = RelayRLAction::from_tensors(
    Some(&obs),
    Some(&act),
    None,          // mask
    1.0,           // reward
    None,          // data
    false,         // done
).expect("serialize action");

// Buffer actions in a trajectory (clears when terminal and capacity exceeded)
let mut traj = RelayRLTrajectory::new(1000);
traj.add_action(&step);

// Terminal step example
let terminal = RelayRLAction::from_tensors(None, None, None, 0.0, None, true).unwrap();
traj.add_action(&terminal);
```

License
-------

Apache-2.0

