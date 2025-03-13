import os
import subprocess
import shutil
import sys
import pytest
from pathlib import Path
from venv import EnvBuilder
import logging
from packaging import version

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def test_experiment_runs(experiment_dir, test_args):
    """Tests if a experiment runs for at least N seconds without errors."""
    # time that device is waiting before timing out, set for RVC4 tests
    os.environ["DEPTHAI_SEARCH_TIMEOUT"] = "30000"

    if test_args["virtual_display"]:
        setup_virtual_display()

    experiment_dir = experiment_dir.resolve()

    dai_experimental = test_args["depthai_version"] == "experimental"
    if dai_experimental and not is_experimental(
        experiment_dir=experiment_dir,
        experimental_subset=test_args["experiments_metadata"]["experimental_subset"],
    ):
        pytest.skip(
            f"Skipping {experiment_dir}: Not part of the experimental DAI subset."
        )

    success, reason = is_valid(
        experiment_dir=experiment_dir,
        known_failing_experiments=test_args["experiments_metadata"][
            "known_failing_experiments"
        ],
        desired_platform=test_args["platform"],
        desired_py=test_args["python_version"],
        desired_dai=""
        if dai_experimental
        else test_args[
            "depthai_version"
        ],  # if experimental DAI then don't check DAI version
    )
    if not success:
        pytest.skip(f"Skipping {experiment_dir}: {reason}")

    main_script = experiment_dir / "main.py"
    requirements_file = experiment_dir / "requirements.txt"
    if not main_script.exists():
        pytest.skip(f"Skipping {experiment_dir}, no main.py found.")
    if not requirements_file.exists():
        pytest.skip(f"Skipping {experiment_dir}, no requirements.txt found.")

    if not dai_experimental:
        venv_dir = experiment_dir / ".test-venv"
        env_exe = venv_dir / "bin" / "python3"

        setup_virtual_env(venv_dir, requirements_file, test_args["depthai_version"])

        success = run_experiment(
            env_exe,
            experiment_dir,
            test_args,
        )
        shutil.rmtree(venv_dir, ignore_errors=True)
        assert success, f"Test failed for {experiment_dir}"
    else:
        # TODO: Implement creating a python environment that has an experimental DAI installed and runns experiment inside it
        raise NotImplementedError(
            "Using DAI `experimental` version is not yet fully supported..."
        )


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
            logger.error(
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

    env = os.environ.copy()
    if env_vars:
        env_dict = dict(item.split("=") for item in env_vars.split())
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


def is_experimental(experiment_dir, experimental_subset):
    """Checks if the experiment is part of the subset of experiments used for experimental DAI version testing"""
    for exp in experimental_subset:
        if exp in str(experiment_dir):
            return True
    return False


def is_valid(
    experiment_dir, known_failing_experiments, desired_platform, desired_py, desired_dai
):
    """Checks if the experiment is valid or known to fail with this parameters.
    If it is known to fail it returns the reason.
    """
    for exp in known_failing_experiments:
        if exp in str(experiment_dir):
            failing_platform = known_failing_experiments[exp].get("platform", None)
            failing_python = known_failing_experiments[exp].get("python_version", None)
            failing_dai = known_failing_experiments[exp].get("depthai_version", None)

            platform_failed = None
            if failing_platform is not None:
                platform_failed = not check_platform(desired_platform, failing_platform)

            python_failed = None
            if failing_python is not None:
                python_failed = not check_platform(desired_py, failing_python)

            dai_failed = None
            if failing_dai is not None:
                dai_failed = not check_platform(desired_dai, failing_dai)

            # Return False only if all checks failed and exclude non relevant checks
            failed = [
                f for f in [platform_failed, python_failed, dai_failed] if f is not None
            ]
            if all(f is True for f in failed):
                if platform_failed:
                    logger.info(
                        f"Platform check failed: Got `{desired_platform}`, shouldn't be `{known_failing_experiments[exp]['platform']}`"
                    )
                if python_failed:
                    logger.info(
                        f"Python version check failed: Got `{desired_py}`, shouldn't be `{known_failing_experiments[exp]['python_version']}`"
                    )
                if dai_failed:
                    logger.info(
                        f"DepthAI version check failed: Got `{desired_dai}`, shouldn't be `{known_failing_experiments[exp]['depthai_version']}`"
                    )
                return (False, known_failing_experiments[exp]["reason"])

    return (True, "")


def check_platform(have, failing):
    if failing == "all":
        return False
    return have not in failing


def check_python(have, failing):
    if failing == "all":
        return False
    return have not in failing


def check_dai(have, failing):
    if have is None or have == "":
        # if not explicitly set we assume it should pass with one specified in requirements
        return True

    if failing == "all":
        return False

    have_version = version.parse(have)

    # Extract operator and version number
    operators = ["<=", ">=", "<", ">"]
    for op in operators:
        if failing.startswith(op):
            version_number = failing[len(op) :]  # Remove operator from string
            failing_version = version.parse(version_number)

            # Perform the appropriate comparison
            if op == "<":
                return not (have_version < failing_version)
            elif op == "<=":
                return not (have_version <= failing_version)
            elif op == ">":
                return not (have_version > failing_version)
            elif op == ">=":
                return not (have_version >= failing_version)

    # If no operator is found, assume exact match
    return not (have_version == version.parse(failing_version))


def get_installed_packages(env_exe):
    """Returns the list of installed packages in the virtual environment."""
    return subprocess.check_output([env_exe, "-m", "pip", "freeze"], text=True)


if __name__ == "__main__":
    pytest.main([__file__])
