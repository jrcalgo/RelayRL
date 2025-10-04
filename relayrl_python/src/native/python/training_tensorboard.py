import asyncio
import json
import logging
import sys
import time
import queue
import threading
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from relayrl_framework import ConfigLoader

from utils.plot import get_newest_dataset

# Configure logging to output INFO level messages to stdout.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class TensorboardWriter:
    """
    Synchronous Tensorboard writer.

    Creates a dedicated thread to write scalar data to Tensorboard based on input
    from a queue. It monitors a progress file (progress.txt) for new scalar data and
    writes these values to Tensorboard.
    
    The TensorboardWriter currently supports writing scalar values only.

    To launch tensorboard manually, use the launch_tensorboard(logdir) function or run:
        tensorboard --logdir <path_to_tensorboard_logs>
    """
    def __init__(self, config_path: str, env_dir=os.getcwd(), algorithm_name: str = 'run'):
        # Load tensorboard configuration parameters
        config = ConfigLoader(config_path=config_path)
        tb_params = config.get_tb_params()
        # Extract values
        scalar_tags = tb_params['scalar_tags']
        global_step_tag = tb_params['global_step_tag']

        # Graceful shutdown mechanism.
        self.shutdown_flag = False
        
        # Initialize Tensorboard writer parameters after first scalar value write operation.
        self.launch_tb_on_startup = tb_params['launch_tb_on_startup']

        # Initialize file and directory paths based on the environment.
        self.writer = None
        self._data_log_dir = env_dir + '/logs'
        self._file_root = get_newest_dataset(self._data_log_dir, return_file_root=True)
        self._tb_log_dir = self._file_root + f'/tb_' + algorithm_name.lower() + f'_{int(time.time())}'
        self._file = self._file_root + '/progress.txt'

        # Queue for storing scalar data to be written.
        self.data_queue = queue.Queue()

        # Validate scalar tags from the progress file.
        self.valid_tags = False
        self.scalar_tags = scalar_tags
        self._global_step_tag = global_step_tag
        self._recent_global_step = 0

        # Event signal to stop the Tensorboard writer loop.
        self._loop_stop_signal = threading.Event()
        # Start the tensorboard writer thread.
        self._tb_thread = threading.Thread(target=self._tensorboard_writer_processes)
        self._tb_thread.daemon = False
        self._tb_thread.start()

        logging.info("[TensorboardWriter] Initialized")

    def manually_queue_scalar(self, tag: str, scalar_value: float, global_step: int):
        """
        Manually enqueue a scalar value to be written to Tensorboard.

        :param tag: The scalar tag (name).
        :param scalar_value: The scalar value.
        :param global_step: The corresponding global step (x-axis value).
        """
        self.data_queue.put(('scalar', tag, scalar_value, global_step))

    def shutdown(self, timeout=None):
        """
        Gracefully shuts down the Tensorboard writer.

        Sets the shutdown flag to True, which signals the writer thread to stop
        processing and close the writer. Optionally waits for the thread to complete.

        :param timeout: Maximum time to wait for the thread to complete (in seconds).
                       If None, wait indefinitely. If 0, don't wait.
        :return: True if the thread was successfully joined, False if timeout occurred.
        """
        logging.info("[TensorboardWriter - shutdown] Initiating shutdown...")
        self.shutdown_flag = True
        self._loop_stop_signal.set()

        if timeout is not None or timeout != 0:
            logging.info(f"[TensorboardWriter - shutdown] Waiting for writer thread to complete (timeout: {timeout}s)...")
            self._tb_thread.join(timeout=timeout)
            thread_completed = not self._tb_thread.is_alive()

            if thread_completed:
                logging.info("[TensorboardWriter - shutdown] Writer thread has completed successfully.")
            else:
                logging.warning("[TensorboardWriter - shutdown] Timeout reached, writer thread still running.")

            return thread_completed
        else:
            logging.info("[TensorboardWriter - shutdown] Shutdown signal sent, not waiting for thread completion.")
            return None

    def _tensorboard_writer_processes(self):
        """
        Main loop for the Tensorboard writer thread.

        This loop validates that required scalar tags exist, retrieves new scalar data from the
        progress file, writes the scalar values to Tensorboard, and flushes the writer.
        It stops when the shutdown_flag is set to True.
        """
        def _validate_tag_existence():
            """
            Validates that the scalar tags exist in the progress.txt file.

            :return: False if the directory, file, or data is missing; None otherwise.
            """
            logging.info("[TensorboardWriter - _validate_tag_existence] Validating scalar tags...")

            if not os.path.exists(self._file_root):
                logging.info("[TensorboardWriter - _validate_tag_existence] Data directory not found. Tensorboard not started.")
                return False

            if not os.path.exists(self._file):
                logging.info("[TensorboardWriter - _validate_tag_existence] progress.txt not found. Tensorboard not started.")
                return False

            if os.path.getsize(self._file) == 0:
                logging.info("[TensorboardWriter - _validate_tag_existence] progress.txt is empty. Tensorboard not started.")
                return False

            data = pd.read_table(self._file)
            if not data.empty:
                for scalar in self.scalar_tags:
                    if scalar not in data.columns:
                        logging.info(f"[TensorboardWriter - _validate_tag_existence] {scalar} not found in progress.txt. Removing from scalar tags.")
                        self.scalar_tags.remove(scalar)
                if not self.scalar_tags:
                    logging.info("[TensorboardWriter - _validate_tag_existence] No scalar tags found. Tensorboard not started.")
                    self.valid_tags = False
                    return None
                else:
                    self.valid_tags = True
                    return None
            else:
                logging.info("[TensorboardWriter - _validate_tag_existence] Data is empty. Tensorboard not started.")
                return False

        def _retrieve_and_queue_data(_previous_last_step: int):
            """
            Retrieves new scalar data from progress.txt and queues it for writing.

            It iterates over the progress file rows starting after the last processed step and enqueues
            scalar values for each scalar tag.

            :param _previous_last_step: The last step index that was processed.
            :return: A tuple (new_last_step, queued_count) where new_last_step is the latest step index processed
                     and queued_count is the number of entries enqueued.
            """
            logging.info("[TensorboardWriter - _retrieve_and_queue_data] Retrieving data from progress.txt...")

            if not os.path.exists(self._file_root):
                logging.info("[TensorboardWriter - _retrieve_and_queue_data] Data directory not found. Tensorboard not started.")
                return _previous_last_step, 0

            if not os.path.exists(self._file):
                logging.info("[TensorboardWriter - _retrieve_and_queue_data] progress.txt not found. Tensorboard not started.")
                return _previous_last_step, 0

            if os.path.getsize(self._file) == 0:
                logging.info("[TensorboardWriter - _retrieve_and_queue_data] progress.txt is empty. Tensorboard not started.")
                return _previous_last_step, 0

            data = pd.read_table(self._file)
            _queued_count = 0
            if not data.empty:
                # Get latest step index
                try:
                    new_last_step = int(data[self._global_step_tag].idxmax())
                except (KeyError, ValueError):
                    logging.warning(f"[TensorboardWriter - _retrieve_and_queue_data] Could not find {self._global_step_tag} in data.")
                    return _previous_last_step, 0

                # Queue new data for each scalar tag
                for scalar in self.scalar_tags:
                    for i in range(_previous_last_step + 1, new_last_step + 1):
                        try:
                            self.data_queue.put(('scalar', scalar, data[scalar][i], data[self._global_step_tag][i]))
                            _queued_count += 1
                        except (KeyError, IndexError):
                            logging.warning(f"[TensorboardWriter - _retrieve_and_queue_data] Error accessing data for {scalar} at step {i}.")
                            continue

                return new_last_step, _queued_count
            else:
                logging.info("[TensorboardWriter - _retrieve_and_queue_data] Data is empty. Tensorboard not started.")
                return _previous_last_step, 0

        # Validate scalar tags until a valid set is found.
        while not self.shutdown_flag:
            if _validate_tag_existence() is not None:
                time.sleep(5)
            else:
                break

        # Check shutdown flag before continuing
        if self.shutdown_flag:
            logging.info("[TensorboardWriter - _tensorboard_processes] Shutdown requested during validation. Stopping.")
            self._loop_stop_signal.set()
            return

        if not self.valid_tags:
            logging.info("[TensorboardWriter - _tensorboard_processes] No valid tags found. Stopping.")
            self._loop_stop_signal.set()
            return
        else:
            # Initialize the Tensorboard SummaryWriter.
            self.writer = SummaryWriter(log_dir=self._tb_log_dir, filename_suffix='_tb')
            previous_last_step = 0

            # Main processing loop - continue until shutdown_flag is True
            while not self.shutdown_flag:
                previous_last_step, queued_count = _retrieve_and_queue_data(previous_last_step)

                if queued_count > 0:
                    try:
                        for count in range(queued_count):
                            # Check for shutdown during processing
                            if self.shutdown_flag:
                                break

                            write_type, *args = self.data_queue.get()
                            if write_type == 'scalar':
                                tag, scalar_value, global_step = args
                                logging.info(f"[TensorboardWriter - _tensorboard_processes] Writing scalar: {args}")
                                self.writer.add_scalar(tag, scalar_value, global_step)
                                self._recent_global_step += 1
                                if self._recent_global_step == 1 and self.launch_tb_on_startup:
                                    # Try to run `tensorboard --logdir logs` command in a separate process
                                    # logdir is presumably found in environment directory, so check there.
                                    launch_tensorboard(self._tb_log_dir)

                        # Only flush if we haven't been asked to shut down
                        if not self.shutdown_flag:
                            self.writer.flush()
                    except queue.Empty:
                        continue
                else:
                    # Use a shorter sleep with periodic shutdown checks
                    for _ in range(10):
                        if self.shutdown_flag:
                            break
                        time.sleep(1)

            # Clean up when shutting down
            logging.info("[TensorboardWriter - _tensorboard_processes] Shutdown flag detected. Stopping writer.")
            if self.writer:
                self.writer.close()
            return


def launch_tensorboard(logdir: str):
    """
    Launches a Tensorboard process to visualize logs.

    Uses the provided log directory for Tensorboard logs.

    If an error occurs while starting Tensorboard, it logs the error.

    :param logdir: Directory containing Tensorboard log files.
    """
    import subprocess
    try:
        logging.info("[launch_tensorboard] Starting Tensorboard.")
        logdir = os.path.join(logdir, 'logs')
        subprocess.run(["tensorboard", "--logdir", logdir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logging.info(f"[launch_tensorboard] Error: {e}")
    finally:
        logging.info("[launch_tensorboard] Tensorboard dashboard unable to start. Please start it manually using `tensorboard --logdir <log_dir>`.")


async def main():
    """
    Main function to initialize tensorboard writer process and pipe output back to Rust backend's PythonRequestChannel.

    :return: stdout messages of tensorboard writer operations.
    """
    import argparse
    # Set up argument parsing for the worker.
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_dir", type=str, default='<directory_of_environment>')
    parser.add_argument("--config_path", type=str, default='<path_to_config_json>')
    parser.add_argument("--algorithm_name", type=str, default='<algorithm_name>')
    args = parser.parse_args()

    # Initialize the TensorboardWriter with provided arguments.
    tb_writer = TensorboardWriter(
        config_path=args.config_path,
        env_dir=args.env_dir,
        algorithm_name=args.algorithm_name
    )

    # Set up an asynchronous stream reader to display output from the TensorboardWriter process.
    # TensorboardWriter writes to stdout and does not accept stdin.
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=5120 * 5120) # 5MB limit
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    # Main loop to continuously process incoming commands.
    while not tb_writer.shutdown_flag:
        line = await reader.readline()
        if not line:
            break

        try:
            command_json = json.loads(line.decode().strip())
            command = command_json.get("command")
            if command == "shutdown":
                tb_writer.shutdown(timeout=10)
            else:
                logging.info(f"[TensorboardWriter - main] Unknown command: {command}")
        except json.JSONDecodeError:
            logging.info(f"[TensorboardWriter - main] Invalid JSON received: {line.decode().strip()}")

        # Forward output to Rust backend
        print(line.decode('utf-8'), end='', flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
