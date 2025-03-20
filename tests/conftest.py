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
        default="",
        help="Specify a depthai version to override requirements.txt.",
    )
    parser.addoption(
        "--depthai-nodes-version",
        type=str,
        default="",
        help="Specify a depthai-nodes version to override requirements.txt. Can be either released version or branch from GH.",
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
        default="rvc2",
        type=str,
        choices=["rvc2", "rvc4"],
        help="Specify a platform this is tested on (rvc2 or rvc4). Only used for filtering test examples.",
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
        "depthai_nodes_version": request.config.getoption("--depthai-nodes-version"),
        "environment_variables": request.config.getoption("--environment-variables"),
        "virtual_display": request.config.getoption("--virtual-display"),
        "platform": request.config.getoption("--platform"),
        "python_version": request.config.getoption("--python-version"),
    }
    logger.info(f"Test arguments: {args}")

    script_dir = Path(__file__).parent
    file_path = script_dir / "experiments_metadata.json"

    with open(file_path) as f:
        experiments_metadata = json.load(f)

    args["experiments_metadata"] = experiments_metadata

    return args


def pytest_generate_tests(metafunc):
    if "experiment_dir" in metafunc.fixturenames:
        root_dir = metafunc.config.getoption("--root-dir")
        exp_dirs = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "main.py" in filenames and "requirements.txt" in filenames:
                exp_dirs.append(Path(dirpath))
            elif "main.py" in filenames and "requirements.txt" not in filenames:
                logger.error(f"Skipping {dirpath} because it has no requirements.txt")
            elif "main.py" not in filenames and "requirements.txt" in filenames:
                logger.error(f"Skipping {dirpath} because it has no main.py")

        metafunc.parametrize("experiment_dir", exp_dirs, ids=[str(p) for p in exp_dirs])
