import os
import subprocess
import shutil
import pytest
import time
from pathlib import Path
import logging
import threading
import queue
import re
import json

from utils import adjust_requirements, is_valid, change_and_restore_dir

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


APP_ID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True, scope="module")
def skip_if_not_rvc4(test_args):
    if test_args["platform"] != "rvc4":
        pytest.skip("Skipping test_standalone.py: requires RVC4 platform")


def test_example_runs_in_standalone(example_dir, test_args):
    """Tests if the example runs in standalone mode for at least N seconds without errors."""
    # Time that device is waiting before timing out, set for RVC4 tests
    os.environ["DEPTHAI_SEARCH_TIMEOUT"] = "30000"

    example_dir = example_dir.resolve()

    success, reason = is_valid(
        example_dir=example_dir,
        known_failing_examples=test_args["examples_metadata"]["known_failing_examples"],
        desired_platform="RVC4",
        desired_py=test_args["python_version"],
        desired_dai=test_args["depthai_version"],
    )
    if not success:
        pytest.skip(f"Skipping {example_dir}: {reason}")

    main_script = example_dir / "main.py"
    requirements_path = example_dir / "requirements.txt"
    oakapp_toml = example_dir / "oakapp.toml"
    if not main_script.exists():
        pytest.skip(f"Skipping {example_dir}, no main.py found.")
    if not requirements_path.exists():
        pytest.skip(f"Skipping {example_dir}, no requirements.txt found.")
    if not oakapp_toml.exists():
        pytest.skip(f"Skipping {example_dir}, no oakapp.toml found.")

    setup_env(
        base_dir=example_dir,
        requirements_path=requirements_path,
        depthai_version=test_args["depthai_version"],
        depthai_nodes_version=test_args["depthai_nodes_version"],
    )

    with change_and_restore_dir(example_dir):
        time.sleep(10)  # to stabilize device
        success = run_example(example_dir=example_dir, args=test_args)
        teardown()

    assert success, f"Test failed for {example_dir}"


def setup_env(
    base_dir: Path,
    requirements_path: Path,
    depthai_version: str | None,
    depthai_nodes_version: str | None,
):
    """Sets up the envrionment with the new requirements"""
    new_requirements = adjust_requirements(
        current_req_path=requirements_path,
        depthai_version=depthai_version,
        depthai_nodes_version=depthai_nodes_version,
    )
    # Create a copy of the old requirements
    shutil.copyfile(requirements_path, base_dir / "requirements_old.txt")
    # Save new requirements
    new_req_path = base_dir / "requirements.txt"
    with open(new_req_path, "w") as f:
        f.writelines(new_requirements)


def enqueue_output(out, q):
    for line in iter(out.readline, ""):
        q.put(line)
    out.close()


def run_example(example_dir: Path, args: dict) -> bool:
    oakctl_path = shutil.which("oakctl")
    assert oakctl_path is not None, "'oakctl' command is not available in PATH"

    connect_timeout = 60
    try:
        result = subprocess.run(
            ["oakctl", "connect", args["device"]],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=connect_timeout,
        )
        device_info = re.sub(r"\s+", " ", result.stdout.decode().strip())
        logger.debug(f"Connected to device: {device_info}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to connect to device `{args['device']}`: {e}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"Timeout ({connect_timeout}s) while trying to connect to device `{args['device']}`"
        )
        return False

    run_duration = args.get("timeout")
    startup_timeout = (
        60 * 5
    )  # if it takes more than 5min to setup the app then fail the test
    try:
        logger.debug(f"Installing {example_dir} app")

        process = subprocess.Popen(
            ["oakctl", "app", "run", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        app_started = False
        start_time = None
        signal_start = time.time()

        for line in process.stdout:
            line = line.strip()
            logger.debug(f"[app output]: {line}")

            # Look for build error in logs
            if "BuilderError" in line:
                process.terminate()
                logger.error(f"Error during build: {line}")
                return False

            # Detect app start trigger
            if "App output:" in line:
                app_started = True
                start_time = time.time()
                logger.info("App start detected. Starting run timer.")
                break

            # Timeout waiting for app to start
            if time.time() - signal_start > startup_timeout:
                process.terminate()
                logger.error(f"Timeout waiting for app start after {startup_timeout}s.")
                return False

        # Setup threading to keep reading app outputs
        q = queue.Queue()
        t = threading.Thread(
            target=enqueue_output, args=(process.stdout, q), daemon=True
        )
        t.start()

        passed = True
        while True and app_started:
            try:
                line = q.get_nowait()
                logger.debug(f"[app output]: {line.strip()}")
            except queue.Empty:
                pass

            status = get_app_status(APP_ID)
            # When app has started, check if it exited early
            if status != "running":
                logger.error(
                    f"App status switched to '{status}' after {time.time() - start_time:.2f}s but should run for {run_duration}s."
                )
                passed = False
                break

            if time.time() - start_time >= run_duration:
                logger.info(f"App ran for {run_duration} seconds successfully.")
                break

            time.sleep(1)

        # Clean up process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        if passed:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Error running app: {e}")
        return False


def get_app_status(app_id: str):
    try:
        result = subprocess.run(
            ["oakctl", "app", "list", "--format=json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        apps = json.loads(result.stdout)
        for app in apps:
            if app["container_id"] == app_id:
                return app["status"]
        return None  # App not found
    except Exception as e:
        logger.warning(f"Failed to query app status: {e}")
        return None


def teardown():
    """Cleans up everything after the test"""
    # Clean up requirements.txt
    if os.path.exists("requirements.txt"):
        os.remove("requirements.txt")
        logger.debug("Deleted requirements.txt")
    if os.path.exists("requirements_old.txt"):
        os.rename("requirements_old.txt", "requirements.txt")
        logger.debug("Renamed requirements_old.txt → requirements.txt")

    # Delete app on device
    try:
        result = subprocess.run(
            ["oakctl", "app", "delete", APP_ID],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.debug(f"App deleted:\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to delete app:\n{e.stderr.strip()}")


if __name__ == "__main__":
    pytest.main([__file__])
