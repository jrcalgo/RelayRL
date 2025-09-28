# RelayRL Framework

**Core Rust Library & Python Bindings for RelayRL**

---

> **Platform Support:**
> RelayRL runs on **MacOS, Linux, and Windows** (x86_64). Some dependencies may require additional setup on Windows.

> **Warning:**  
> This is a **prototype** and is **unstable during training**.  
> For general usage and project overview, see the [root README](../README.md).

---

## Overview

The `relayrl_framework` crate provides the core infrastructure for distributed RL experiments, including:

- High-performance agent and training server implementations (gRPC & ZMQ)
- Core RL data structures (actions, trajectories, configs)
- Python bindings via PyO3 for seamless integration with Python RL algorithms
- Utilities for configuration, logging, and benchmarking

---

**Default Rust Usage Example**

```rust
use relayrl_framework::network::client::agent_wrapper::RelayRLAgent;
use relayrl_framework::network::server::training_server_wrapper::TrainingServer;
use std::path::PathBuf;
use tch::{CModule, Tensor};
use tokio;

#[tokio::main]
async fn main() {
    // Start a training server (REINFORCE, discrete, CartPole)
    let _server = TrainingServer::new(
        "REINFORCE".to_string(), // algorithm_name
        4,                        // obs_dim
        2,                        // act_dim
        100000,                   // buf_size
        false,                    // tensorboard
        false,                    // multiactor
        Some("./env".to_string()),
        None,                     // algorithm_dir
        Some(PathBuf::from("relayrl_config.json")),
        None,                     // hyperparams
        Some("zmq".to_string()), // server_type
        None, None, None          // training_prefix, training_host, training_port
    ).await;

    // Load a TorchScript model (optional, can be None to fetch from server)
    let model = None; // Or: Some(CModule::load("client_model.pt").expect("Failed to load model"));

    // Create a ZMQ agent (use "grpc" for gRPC)
    let agent = RelayRLAgent::new(
        model,
        Some(PathBuf::from("relayrl_config.json")),
        Some("zmq".to_string()),
        None, // training_prefix
        None, // training_port
        None, // training_host
    ).await;

    // Example: Request an action (replace with your actual observation and mask tensors)
    // let obs = Tensor::of_slice(&[0.0, 0.1, 0.0, 0.0]).reshape(&[1, 4]);
    // let mask = Tensor::of_slice(&[1.0, 1.0]).reshape(&[1, 2]);
    // let reward = 0.0_f32;
    // let action = agent.request_for_action(obs, mask, reward).await;
}
```

**Default Python Usage Example**

```python
from relayrl_framework import TrainingServer, RelayRLAgent

# Start a training server (REINFORCE, discrete, CartPole)
server = TrainingServer(
    algorithm_name="REINFORCE",
    obs_dim=4,           # CartPole observation space
    act_dim=2,           # CartPole action space
    buf_size=100000,
    config_path="relayrl_config.json",
    server_type="zmq"
)

# Create an agent and interact with the environment
agent = RelayRLAgent(
    model_path=None,                 # None to fetch model from server
    config_path="relayrl_config.json",
    server_type="zmq"
)

# Example: Request an action from the agent
obs = [0.0, 0.1, 0.0, 0.0]  # Should be a numpy array or torch tensor in real use
mask = [1.0, 1.0]           # Should be a numpy array or torch tensor
reward = 0.0
action_obj = agent.request_for_action(obs, mask, reward)
action = action_obj.get_act()
```

## More Explanatory Tutorial

[See the examples' README!](https://github.com/jrcalgo/RelayRL-proto/blob/main/examples/README.md#how-to-use-in-novel-environments)

---

**Framework Requirements:**
- **Python Algorithms:** The framework requires Python RL algorithms to be implemented and provided
- **TorchScript Models:** Algorithms must export their models as TorchScript for deployment
- **Custom Algorithm Support:** Custom algorithms can be integrated via configuration parameters

---

## Building & Development

### Prerequisites

- **Supported Platforms:** MacOS, Linux, Windows (x86_64)
- Rust (latest stable recommended)
- Python 3.8+
- PyTorch 2.5.1
- [maturin](https://github.com/PyO3/maturin) (for building Python bindings)
- Protobuf compiler (`protoc`) for gRPC

### Build the Rust Library

```sh
cargo build --release
```

### Build & Install Python Bindings

From the `relayrl_framework` directory:

```sh
pip install maturin
maturin develop
```

This will build the Rust library and install the Python bindings into your current environment.

---

## Python Algorithms

### Included Algorithms

Currently, the framework includes:
- **REINFORCE with baseline value functions**
- **REINFORCE without baseline value functions**

### Custom Algorithm Implementation

Custom algorithms can be implemented and integrated into the framework. Your algorithm must:

1. **Inherit from BaseAlgorithm:** Extend the `AlgorithmAbstract` class
2. **Export Required Methods:** Use `@torch.jit.export` annotations for critical methods
3. **Implement Core Interface:** Provide `step`, `get_obs_dim`, and `get_act_dim` methods

#### Required TorchScript Exports

Your algorithm's model must export these methods via `@torch.jit.export`:

```python
import torch
import torch.nn as nn

class CustomPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.input_dim = obs_dim
        self.output_dim = act_dim
        # Your network architecture here
        
    @torch.jit.export
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        """Execute one step of the policy"""
        with torch.no_grad():
            # Your inference logic here
            action = self.forward(obs, mask)
            logp_a = self.get_log_prob(obs, action)
        data = {'logp_a': logp_a}
        return action, data
        
    @torch.jit.export
    def get_obs_dim(self):
        """Return observation dimension"""
        return self.input_dim
        
    @torch.jit.export
    def get_act_dim(self):
        """Return action dimension"""
        return self.output_dim
```

#### Algorithm Class Structure

```python
from _common._algorithms.BaseAlgorithm import AlgorithmAbstract
from relayrl_framework import RelayRLTrajectory, ConfigLoader

class CustomAlgorithm(AlgorithmAbstract):
    def __init__(self, env_dir: str, config_path: str, obs_dim: int, act_dim: int, buf_size: int):
        super().__init__()
        
        # Load configuration
        config_loader = ConfigLoader(algorithm_name='CUSTOM_ALGO', config_path=config_path)
        hyperparams = config_loader.get_algorithm_params()['CUSTOM_ALGO']
        self.save_model_path = config_loader.get_server_model_path()
        
        # Initialize your model
        self._model = CustomPolicy(obs_dim, act_dim)
        
    def save(self) -> None:
        """Save model as TorchScript"""
        self._model.eval()
        model_script = torch.jit.script(self._model)
        torch.jit.save(model_script, self.save_model_path)
        self._model.train()
        
    def receive_trajectory(self, trajectory: RelayRLTrajectory) -> bool:
        """Process received trajectory and return True if training should occur"""
        # Your training logic here
        return training_ready
```

### Using Custom Algorithms

#### Configuration

Specify custom algorithms in your `relayrl_config.json`:

```json
{
    "algorithms": {
        "CUSTOM_ALGO": {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99
        }
    },
    "server": {
        "training_server": {
            "host": "127.0.0.1",
            "port": "50051"
        }
    }
}
```

#### Server Initialization

```python
from relayrl_framework import TrainingServer

# Initialize server with custom algorithm
server = TrainingServer(
    algorithm_name="CUSTOM_ALGO",
    algorithm_dir="/path/to/your/algorithm",
    obs_dim=8,
    act_dim=4,
    buf_size=1000000,
    config_path="relayrl_config.json",
    server_type="ZMQ"
)
```

#### Agent Initialization

```python
from relayrl_framework import RelayRLAgent

# Initialize agent with custom algorithm
agent = RelayRLAgent(
    algorithm_name="CUSTOM_ALGO",
    config_path="relayrl_config.json",
    server_type="ZMQ"
)
```

---

## Running Benchmarks

Benchmarks are provided in the `benches/` directory:

```sh
cargo bench
```

You can also use the provided shell scripts in `scripts/` for profiling and release builds.

---

## Development Notes

- **Python Bindings:**  
  Expose `RelayRLAction`, `RelayRLTrajectory`, `ConfigLoader`, and agent/server classes to Python.
- **Configuration:**  
  Use JSON files to specify experiment parameters, network settings, and logging options.
- **Algorithm Integration:**  
  Custom algorithms must be properly structured with TorchScript exports and inherit from the base algorithm class.

---

## Limitations

- **Single-Agent Focus:**  
  The framework is designed for single-agent RL. Multi-agent support is not natively implemented, but you may launch multiple agent and training server processes for experimentation.
- **Prototype Status:**  
  The framework is unstable during training and is under active development. Expect breaking changes and incomplete features.
- **Python Algorithm Requirement:**  
  The framework requires Python algorithms to be implemented and provided - it does not include built-in algorithms beyond the basic REINFORCE implementations.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

---

## License

[Apache License 2.0](../LICENSE)
