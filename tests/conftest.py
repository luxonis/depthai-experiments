import pytest
from pathlib import Path
import os
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def pytest_addoption(parser):
    parser.addoption(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the directory with projects or a single project.",
    )
    parser.addoption(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for script execution (default: 30s).",
    )
    parser.addoption(
        "--depthai-version",
        type=str,
        help="Specify a depthai version to override requirements.txt.",
    )
    parser.addoption(
        "--environment-variables",
        type=str,
        help="List of additional environment variables (format: VAR1=VAL1 VAR2=VAL2).",
    )
    parser.addoption(
        "--virtual-display",
        action="store_true",
        help="Enable virtual display (sets DISPLAY=':99').",
    )
    parser.addoption(
        "--platform",
        default="RVC2",
        type=str,
        choices=["RVC2", "RVC4"],
        help="Specify a platform this is tested on (RVC2 or RVC4). Only used for filtering test examples.",
    )
    parser.addoption(
        "--python-version",
        default="3.10",
        type=str,
        choices=["3.8", "3.10", "3.12"],
        help="Specify a python version this is tested with (3.8, 3.10 or 3.12). Only used for filtering test examples.",
    )


@pytest.fixture(scope="session")
def test_args(request):
    args = {
        "root_dir": request.config.getoption("--root-dir"),
        "timeout": request.config.getoption("--timeout"),
        "depthai_version": request.config.getoption("--depthai-version"),
        "environment_variables": request.config.getoption("--environment-variables")
        or {},
        "virtual_display": request.config.getoption("--virtual-display"),
        "platform": request.config.getoption("--platform"),
        "python_version": request.config.getoption("--python-version"),
    }
    return args


def pytest_generate_tests(metafunc):
    if "experiment_dir" in metafunc.fixturenames:
        root_dir = metafunc.config.getoption("--root-dir")
        platform = metafunc.config.getoption("--platform")
        python_version = metafunc.config.getoption("--python-version")
        exp_dirs = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "main.py" in filenames and "requirements.txt" in filenames:
                exp_dirs.append(Path(dirpath))
            elif "main.py" in filenames and "requirements.txt" not in filenames:
                logger.error(f"Skipping {dirpath} because it has no requirements.txt")
            elif "main.py" not in filenames and "requirements.txt" in filenames:
                logger.error(f"Skipping {dirpath} because it has no main.py")

        exp_dirs = filter_experiments(exp_dirs, platform, python_version)
        metafunc.parametrize("experiment_dir", exp_dirs, ids=[str(p) for p in exp_dirs])


def filter_experiments(all_exp_dirs, curr_platform, curr_python_version):
    filtered_exp_dirs = []

    script_dir = Path(__file__).parent
    file_path = script_dir / "known_failing_examples.json"

    with open(file_path) as f:
        data = json.load(f)
    excluded_dirs = (
        data[curr_platform][curr_python_version] + data[curr_platform]["all"]
    )
    for exp_dir in all_exp_dirs:
        logger.info(exp_dir, excluded_dirs)
        if any([str(i) in str(exp_dir) for i in excluded_dirs]):
            logger.error(
                f"Skipping {exp_dir} because not supported on platform={curr_platform} with python={curr_python_version}"
            )
        else:
            filtered_exp_dirs.append(exp_dir)
    return filtered_exp_dirs
