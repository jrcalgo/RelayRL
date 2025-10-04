# Python Channel Reply

import sys
import logging

# Configure logging to output INFO level messages to stdout.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import json
import importlib
import asyncio

from relayrl_framework import RelayRLTrajectory


class LoadScripts:
    """
    LoadScripts

    This class is responsible for dynamically loading the algorithm module based on the 
    provided algorithm name and hyperparameters, and initializing the algorithm instance.
    """
    def __init__(self, algorithm_name, hyperparams, env_dir, resolved_algorithm_dir, config_path):
        # Add the resolved algorithm directory to sys.path so Python can import the module.
        sys.path.append(resolved_algorithm_dir)
        
        # Parse hyperparameters: if hyperparams are provided as a JSON string, load them into a dict.
        hyperparams = json.loads(hyperparams) if hyperparams else None
        # Convert numeric strings to numbers.
        if hyperparams is not None:
            for k, v in hyperparams.items():
                if isinstance(v, str) and v.isdigit():
                    if "." in v:
                        hyperparams[k] = float(v)
                    else:
                        hyperparams[k] = int(v)
        # append config path
        hyperparams['config_path'] = config_path

        # Construct the module name and import it.
        algorithm_module = f"{algorithm_name}.{algorithm_name}"
        algorithm_module_import = importlib.import_module(algorithm_module)
        # Get the algorithm class from the module.
        algorithm_class = getattr(algorithm_module_import, algorithm_name)
        # Initialize the algorithm instance with hyperparameters if available.
        self.algorithm = algorithm_class(**hyperparams) if algorithm_class else None

        # Log critical status based on whether the algorithm was successfully initialized.
        if self.algorithm is None:
            logging.critical("[algorithm_pyscript_false]")
        else:
            logging.critical("[algorithm_pyscript_true]")


class PythonWorker:
    """
    PythonWorker

    This class handles commands received from the Rust PythonCommandCenter.
    It processes commands asynchronously and sends back JSON responses.
    """
    def __init__(self, algorithm_name, hyperparams, env_dir, resolved_algorithm_dir, config_path):
        # Load the algorithm using LoadScripts.
        self.loaded_scripts = LoadScripts(algorithm_name, hyperparams, env_dir, resolved_algorithm_dir, config_path)
        # An asyncio lock to ensure that commands are handled serially.
        self.lock = asyncio.Lock()
        # Flag to signal when to shut down the worker.
        self.shutdown_flag = False

    async def save_model(self):
        """
        Asynchronously instructs the algorithm to save its current model.
        
        Returns:
            A JSON-compatible dict indicating success or error.
        """
        async with self.lock:
            if self.loaded_scripts.algorithm:
                try:
                    # Run the save operation in a thread to avoid blocking the event loop.
                    await asyncio.to_thread(self.loaded_scripts.algorithm.save)
                    return {"status": "success"}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": "Algorithm not initialized"}

    async def receive_trajectory(self, trajectory):
        """
        Asynchronously processes a received trajectory.
        
        Converts a JSON trajectory into an RelayRLTrajectory object, then passes it
        to the algorithm's receive_trajectory method.
        
        Returns:
            A JSON-compatible dict indicating success, error, or that no update occurred.
        """
        # Convert the JSON trajectory into an RelayRLTrajectory object in a separate thread.
        trajectory_obj = await asyncio.to_thread(RelayRLTrajectory.traj_from_json, trajectory)
        async with self.lock:
            if self.loaded_scripts.algorithm:
                try:
                    # Process the trajectory and get the update status.
                    status = await asyncio.to_thread(self.loaded_scripts.algorithm.receive_trajectory, trajectory_obj)
                    if status:
                        return {"status": "success"}
                    else:
                        return {"status": "not_updated"}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": "Algorithm not initialized"}

    async def handle_command(self, command_json):
        """
        Handles a single command represented as a JSON dict.
        
        Supported commands:
        - "save_model": Triggers model saving.
        - "receive_trajectory": Processes a trajectory.
        
        Returns:
            A JSON-compatible dict with the result of the command.
        """
        command = command_json.get("command")
        if command == "save_model":
            return await self.save_model()
        elif command == "receive_trajectory":
            trajectory = command_json.get("trajectory")
            return await self.receive_trajectory(trajectory)
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}


async def main():
    import argparse
    # Set up argument parsing for the worker.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm_name", type=str, default=None)
    parser.add_argument("--env_dir", type=str, default=None)
    parser.add_argument("--resolved_algorithm_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--hyperparams", type=str, default=None)
    args = parser.parse_args()

    # Initialize the PythonWorker with provided arguments.
    worker = PythonWorker(
        algorithm_name=args.algorithm_name,
        hyperparams=args.hyperparams,
        env_dir=args.env_dir,
        resolved_algorithm_dir=args.resolved_algorithm_dir,
        config_path=args.config_path,
    )

    # Set up an asynchronous stream reader to process commands from stdin.
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=10240 * 10240) # 10MB limit
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    # Main loop to continuously process incoming commands.
    while not worker.shutdown_flag:
        line = await reader.readline()
        if not line:
            break

        try:
            command_json = json.loads(line.decode().strip())
            response = await worker.handle_command(command_json)
        except json.JSONDecodeError:
            response = {"status": "error", "message": "Invalid JSON"}
        except Exception as e:
            response = {"status": "error", "message": str(e)}

        # Output the response as a JSON string.
        print(json.dumps(response))
        sys.stdout.flush()

    print("[PythonWorker] Shutting down worker...")


if __name__ == "__main__":
    asyncio.run(main())
