# RelayRL Example Notebooks / Environment Tutorial

This directory contains example Jupyter notebooks demonstrating how to use the RelayRL framework with different reinforcement learning algorithms, environments, and communication backends (gRPC and ZMQ).

## Directory Structure

```
examples/
  REINFORCE_with_baseline/
    classic_control/
      cartpole/
        grpc/
          cartpole_grpc.ipynb
        zmq/
          cartpole_zmq.ipynb
      mountain_car/
        grpc/
          mountain_car_grpc.ipynb
        zmq/
          mountain_car_zmq.ipynb
    box2d/
      lunar_lander/
        grpc/
          lunar_lander_grpc.ipynb
        zmq/
          lunar_lander_zmq.ipynb
  REINFORCE_without_baseline/
    classic_control/
      cartpole/
        grpc/
          cartpole_grpc.ipynb
        zmq/
          cartpole_zmq.ipynb
      mountain_car/
        grpc/
          mountain_car_grpc.ipynb
        zmq/
          mountain_car_zmq.ipynb
    box2d/
      lunar_lander/
        grpc/
          lunar_lander_grpc.ipynb
        zmq/
          lunar_lander_zmq.ipynb
  requirements.txt
  README.md
```

## How to Use

1. **Install dependencies** as described in the main project README.
2. **Open any notebook** (e.g., `REINFORCE_with_baseline/classic_control/cartpole/grpc/cartpole_grpc.ipynb`) in Jupyter or VSCode.
3. **Run the cells** to launch the training server and start the RL training loop using the RelayRL Python bindings.
4. **Artifacts:**
   - When you run an example, `server_model.pt` and `client_model.pt` are created in the same directory as the notebook (e.g., `REINFORCE_with_baseline/classic_control/cartpole/grpc/`).
   - A `logs/` directory is also created in the same location, containing subdirectories for each run. Each run directory stores:
     - `progress.txt` (training performance and PyTorch algorithm metrics)
     - `config.json` (the deployed config for the run)
     - (Optionally) TensorBoard event files, if enabled in the config.
   - You can point TensorBoard to the `logs/` directory to visualize training if TensorBoard is enabled.
  
## Adding New Example Environments

To add a new environment (e.g., BipedalWalker or CartPole):
- Copy one of the existing notebooks (e.g., `lunar_lander_grpc.ipynb` or `cartpole_zmq.ipynb`).
- Update the `env = gym.make(...)` line to your desired environment.
- Set the correct observation and action dimensions in the `TrainingServer` and `RelayRLAgent` initialization.
- Place the notebook in the appropriate directory reflecting the backend (grpc or zmq), algorithm (PPO or REINFORCE), and environment type (classic_control, box2d, etc).

## Notes

- All examples use the RelayRL Python API, which interfaces with the underlying Rust implementation.
- If you encounter missing dependencies, ensure your Python environment matches the requirements in `relayrl_framework/src/native/python/requirements`.
- The structure supports easy extension to new environments and algorithms.
- Model and log artifacts are always created in designated `env_dir` path variable as the environment you run.
- Server-side logs reflect training performance and other PyTorch algorithm metrics. The deployed config and (optional) TensorBoard files are stored in the logs directory for each run.

## How to Use in Novel Environments

RelayRL is designed to be easily integrated into your own environments or applications, outside of the provided `examples/` directory. Below is a detailed guide for deploying RelayRL in custom or novel environments.

### 1. Install the RelayRL Framework

Follow the installation instructions in the main project README to build and install the Python bindings. Ensure all dependencies are installed in your Python environment.

### 2. Prepare Your Environment

- Implement or select a Gymnasium-compatible environment (or adapt your environment to follow the Gym API).
- Decide on the RL algorithm (e.g., PPO, REINFORCE) and communication backend (ZMQ or gRPC).
- Prepare a configuration file (JSON) if you want to override defaults (optional).

### 3. Launch the Training Server

The `TrainingServer` manages the training process and communicates with agents. It should be launched in its own process or script. Example:

```python
from relayrl_framework import TrainingServer
import gymnasium as gym

# Example: Using CartPole-v1, but replace with your custom environment
env = gym.make('CartPole-v1')

server = TrainingServer(
    algorithm_name="REINFORCE",                # RL algorithm: "REINFORCE" (only one for now)
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.n,
    buf_size=1000000,                     # Replay buffer size
    tensorboard=True,                     # Enable TensorBoard logging
    env_dir="/path/to/your/env_dir",     # Directory for logs, models, etc.
    config_path="/path/to/your/config.json",  # Optional: custom config
    server_type="ZMQ",                   # "ZMQ" or "gRPC"
    # algorithm_dir, hyperparams, training_host, training_port, etc. can also be set
)
```

**Notes:**
- The `env_dir` parameter controls where all model and log artifacts are stored for this environment/run.
- The `config_path` can point to a custom JSON config file. If not provided, defaults are used.
- The server can be run on a different machine from the agent(s), as long as network addresses are configured.

### 4. Launch the Agent(s)

Agents interact with the environment and communicate with the server. Each agent can be run in its own process or script. Example:

```python
from relayrl_framework import RelayRLAgent
import gymnasium as gym
import numpy as np

env = <make('YourCustomEnv-v0')>
agent = RelayRLAgent(
    config_path=<"/path/to/your/config.json">,  # Should match the server config
    server_type="ZMQ"                        # Should match the server
)

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    mask = np.ones(env.action_space.n, dtype=np.float32)  # Action mask (if needed)
    reward = 0.0
    while not done:
        action_obj = agent.request_for_action(obs, mask, reward)
        action_value = int(action_obj.get_act())
        next_obs, reward, terminated, truncated, _ = env.step(action_value)
        done = terminated or truncated
        agent.flag_last_action(reward)
        obs = next_obs
        total_reward += reward
    print(f'Episode {episode+1}: Total Reward = {total_reward}')
env.close()
```

**Key Points:**
- The agent and server must use the same communication backend (`server_type`) and compatible config.
- You can run multiple agents in parallel, each connecting to different servers for distributed or multi-process training.
- The `mask` argument is used for action masking (set to all ones if not needed).

### 5. Artifacts and Logging

- **Model Files:** `server_model.pt` and `client_model.pt` are saved in the `env_dir` you specify.
- **Logs:** A `logs/` directory is created in `env_dir`, with subdirectories for each run containing:
  - `progress.txt` (training performance and metrics)
  - `config.json` (deployed config for the run)
  - (Optionally) TensorBoard event files based off progress.txt, if enabled
- **TensorBoard:** If enabled, you can point TensorBoard to the `logs/` directory for real-time visualization:
  ```sh
  tensorboard --logdir /path/to/your/env_dir/logs
  ```

### 6. Advanced Usage

- **Custom Algorithms:** You can extend RelayRL with your own algorithms by following the structure in `relayrl_framework/src/native/python/algorithms/`.
- **Custom Configs:** Use the `config_path` parameter to point to a custom JSON config for hyperparameters, logging, and backend settings.
- **Distributed Training:** Run the server and agents on different machines by setting appropriate network addresses in the config.
- **Multi-Agent:** Launch multiple agents, all connecting to the different server.

### 7. Troubleshooting

- Ensure your Python environment matches the requirements in `relayrl_framework/src/native/python/requirements`.
- Check that network ports are open and not blocked by firewalls if running across machines.
- Review logs in the `logs/` directory for error messages and training diagnostics.

---

For more details, see the main project [README](../README.md).
