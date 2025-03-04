import os
import subprocess
import shutil
import sys
import pytest
from pathlib import Path
from venv import EnvBuilder
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def test_experiment_runs(experiment_dir, test_args):
    """Tests if a experiment runs for at least N seconds without errors."""
    if test_args["virtual_display"]:
        setup_virtual_display()

    experiment_dir = experiment_dir.resolve()

    main_script = experiment_dir / "main.py"
    requirements_file = experiment_dir / "requirements.txt"
    venv_dir = experiment_dir / ".test-venv"
    env_exe = venv_dir / "bin" / "python3"

    if not main_script.exists():
        pytest.skip(f"Skipping {experiment_dir}, no main.py found.")
    if not requirements_file.exists():
        pytest.skip(f"Skipping {experiment_dir}, no requirements.txt found.")

    setup_virtual_env(venv_dir, requirements_file, test_args["depthai_version"])
    success = run_experiment(
        env_exe,
        experiment_dir,
        test_args,
    )
    shutil.rmtree(venv_dir, ignore_errors=True)
    assert success, f"Test failed for {experiment_dir}"


def setup_virtual_display():
    logger.debug("Ensuring virtual display is set up...")
    result = subprocess.run(
        ["pgrep", "-f", "Xvfb :99"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        logger.debug("Starting virtual display...")
        result = subprocess.run(
            ["which", "Xvfb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            logger.debug(
                "Xvfb is not installed on this machine. Please install it and try again."
            )
            sys.exit(1)
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def setup_virtual_env(venv_dir, requirements_file, depthai_version):
    """Creates and sets up a virtual environment with the required dependencies."""
    logger.debug(f"Setting up virtual environment for {venv_dir.parent}...")
    EnvBuilder(clear=True, with_pip=True).create(venv_dir)
    env_exe = venv_dir / "bin" / "python3"

    # Modify requirements.txt if depthai version is specified
    with open(requirements_file, "r") as f:
        requirements = f.readlines()
    if depthai_version:
        requirements = [
            f"depthai=={depthai_version}\n" if "depthai==" in line else line
            for line in requirements
        ]

    with open(venv_dir / "requirements.txt", "w") as f:
        f.writelines(requirements)

    # Install dependencies
    try:
        subprocess.run(
            [
                env_exe,
                "-m",
                "pip",
                "install",
                "-r",
                str(venv_dir / "requirements.txt"),
                "--timeout=60",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.debug(f"Installed packages:\n{get_installed_packages(env_exe)}")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(venv_dir)
        pytest.fail(f"Failed to install dependencies for {venv_dir.parent}: {e.stderr}")


def run_experiment(env_exe, experiment_dir, args):
    """Runs the main.py script for the given timeout duration."""
    timeout = args["timeout"]
    env_vars = args["environment_variables"]

    virtual_env = args["virtual_display"]
    logger.debug(f"Running {experiment_dir} with timeout {timeout}s...")

    main_script = "main.py"

    original_dir = Path.cwd()
    os.chdir(experiment_dir)

    if env_vars:
        env_dict = dict(item.split("=") for item in env_vars.split())
        env = os.environ.copy()
        env.update(env_dict)

    if virtual_env:
        env["DISPLAY"] = ":99"

    try:
        # Run the experiment script (main.py)
        result = subprocess.run(
            [env_exe, str(main_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=timeout,
        )

        if result.returncode == 0:
            logger.error(
                f"{experiment_dir} ran for {timeout} seconds before terminating (exit code 0)."
            )
            return False
        else:
            logger.error(f"Error in {experiment_dir}:\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.info(f"{experiment_dir} ran for {timeout} seconds before timeout.")
        return True
    finally:
        # Restore the original working directory after running the script
        os.chdir(original_dir)


def get_installed_packages(env_exe):
    """Returns the list of installed packages in the virtual environment."""
    return subprocess.check_output([env_exe, "-m", "pip", "freeze"], text=True)


if __name__ == "__main__":
    pytest.main([__file__])
