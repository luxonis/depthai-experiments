import os
import sys
from venv import EnvBuilder
import subprocess
import shutil
import datetime
import time
import argparse

"""
USAGE: gen3_script_tester.py [-p PATH] [-t TIMEOUT] [-s] [-dv DEPTHAI_VERSION] [-e ENVIRONMENT_VARIABLES]

optional arguments:
  -p PATH, --path PATH         The path to a single directory to be tested (otherwise searches for all valid experiments in gen3)
  -t, --timeout                The time it takes to evaluate a running program as working (in seconds)
  -s, --save                   Saves the output log to a file (otherwise just prints it)
  -dv DV, --depthai-version DV Installs specified depthai version for each experiment
  -e VARS, --env VARS          Environment variables to be passed to the script
  -vd, --virtual-display       Starts a virtual display for the script to run in
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="The path to a single directory to be tested (otherwise searches for all valid experiments in gen3)",
)
parser.add_argument(
    "-t",
    "--timeout",
    type=int,
    default=30,
    help="The time it takes to evaluate a running program as working (in seconds)",
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    help="Saves the output log to a file (otherwise just prints it)",
)
parser.add_argument(
    "-dv",
    "--depthai-version",
    type=str,
    help="Installs specified depthai version for each experiment",
)
parser.add_argument(
    "-e",
    "--environment-variables",
    type=str,
    help="Environment variables to be passed to the script",
)
parser.add_argument(
    "-vd",
    "--virtual-display",
    action="store_true",
    help="Starts a virtual display for the script to run in",
)
args = parser.parse_args()


def output(text, f):
    if args.save:
        f.write(text + "\n")
    print(text)


def setup_venv_exe(dir, requirements_path, f=None):
    env_dir = os.path.join(dir, ".test-venv")
    env_builder = EnvBuilder(clear=True, with_pip=True, system_site_packages=False)
    env_builder.create(env_dir)
    env_bin = os.path.join(env_dir, "bin")
    env_exe = os.path.join(env_bin, "python3")

    with open(requirements_path, "r") as f:
        requirements = f.readlines()
        if args.depthai_version:
            for i, requirement in enumerate(requirements):
                if "depthai==" in requirement:
                    requirements[i] = "depthai==" + args.depthai_version + "\n"
    try:
        script = (
            env_exe
            + " -m pip install --pre "
            + "--extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local/ "
            + "--extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ "
            + "".join(requirements)
        )
        script = script.replace("\n", " ")
        subprocess.run(
            script,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        output("Requirements could not be downloaded", f)
        lines = e.stdout.split("\n")
        error_printed = False
        for line in lines:
            if "error" in line.lower():
                output("Error = " + line, f)
                error_printed = True
        if not error_printed:
            output("Error message:\n" + str(e.stdout), f)
        output("----------------------------", f)
        return None
    else:
        output("Requirements installed", f)
        return env_exe


def test_directory(dir, f=None):
    main = os.path.join(dir, "main.py")
    requirements = os.path.join(dir, "requirements.txt")
    if os.path.isfile(main) and os.path.isfile(requirements):
        output("Testing " + dir, f)
        start_time = time.time()

        executable = setup_venv_exe(dir, requirements, f)
        if executable is None:
            return False

        start_time = time.time()
        script = [executable, main]
        env = os.environ.copy()
        if args.virtual_display:
            env["DISPLAY"] = ":99"
        if args.environment_variables:
            env_vars = dict(
                var.split("=") for var in args.environment_variables.split()
            )
            env.update(env_vars)

        process = subprocess.Popen(
            script, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            output("Main ran successfully for " + str(args.timeout) + " seconds", f)
            success = True
        else:
            if process.returncode != 0:
                duration = time.time() - start_time
                process_output = process.stdout.read()
                lines = process_output.split("\n")
                output("Main terminated after " + str(duration) + " seconds", f)
                error_printed = False
                for line in lines:
                    if "error" in line.lower():
                        output("Error = " + line, f)
                        error_printed = True
                if not error_printed:
                    output("Error message:\n" + process_output, f)
                success = False
            else:
                output(
                    "Main finished successfully under "
                    + str(args.timeout)
                    + " seconds",
                    f,
                )
                success = True
        finally:
            process.kill()  # Ensure the process is killed in the finally block
            process.wait()  # Ensure the process is fully terminated
            shutil.rmtree(os.path.join(dir, ".test-venv"))
            output("----------------------------", f)

    elif os.path.isfile(main) and "pip/_internal" not in dir:
        output("Testing: " + dir, f)
        output("Folder has main but not requirements", f)
        output("----------------------------", f)
        success = False
    elif os.path.isfile(requirements):
        output("Testing: " + dir, f)
        output("Folder has requirements but not main", f)
        output("----------------------------", f)
        success = False
    else:  # Folder without requirements or main -> continue
        success = True
    return success


if args.virtual_display:
    print("Ensuring virtual display is set up...")
    result = subprocess.run(
        ["pgrep", "-f", "Xvfb :99"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print("Starting virtual display...")
        result = subprocess.run(
            ["which", "Xvfb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            print(
                "Xvfb is not installed on this machine. Please install it and try again."
            )
            sys.exit(1)
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1920x1080x24"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
print("Starting test...")

success = True
if args.save:
    log_file = "test_" + datetime.datetime.now().strftime("%H:%M:%S") + ".txt"
    with open(log_file, "w") as f:
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
            success &= test_directory(dirpath, f)
        else:
            success &= test_directory(args.path, f)

        print("Test finished, results in: " + log_file)
else:
    if args.path is None:
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
            success &= test_directory(dirpath)
    else:
        success &= test_directory(args.path)

    print("Test finished " + ("successfully" if success else "with errors"))

sys.exit(0 if success else 1)
