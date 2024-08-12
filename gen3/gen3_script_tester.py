import os
from venv import EnvBuilder
import subprocess
import shutil
import datetime
import time
import argparse

"""
USAGE: gen3_script_tester.py [-p PATH] [-t TIMEOUT] [-s]

optional arguments:
  -p PATH, --path PATH      The path to a single directory to be tested (otherwise searches for all valid experiments in gen3)
  -t, --timeout             The time it takes to evaluate a running program as working (in seconds)
  -s save                   Saves the output log to a file (otherwise just prints it)
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="The path to a single directory to be tested (otherwise searches for all valid experiments in gen3)",
)
parser.add_argument(
    "-t", "--timeout"
    , type=int
    , default=30
    , help="The time it takes to evaluate a running program as working (in seconds)"
)
parser.add_argument(
    "-s", "--save"
    , action="store_true"
    , help="Saves the output log to a file (otherwise just prints it)"
)
args = parser.parse_args()

def output(text, f):
    if args.save:
        f.write(text+"\n")
    print(text)

def setup_venv_exe(dir, f=None):
    env_dir = os.path.join(dir, ".test-venv")
    env_builder = EnvBuilder(clear=True, with_pip=True, system_site_packages=False)
    env_builder.create(env_dir)
    env_bin = os.path.join(env_dir, 'bin')
    env_exe = os.path.join(env_bin, 'python3')

    try:
        subprocess.run(env_exe + " -m pip install -r requirements.txt -r requirements.txt --pre"
                   " --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local/"
                   , shell=True, cwd=dir, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output("Requirements could not be downloaded", f)
        for line in e.stdout.split("\n"):
            if "error" in line.lower():
                output("Error = " + line, f)
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

        try:
            executable = setup_venv_exe(dir, f)
            if executable is None:
                return

            start_time = time.time()
            subprocess.run(executable + " " + main, shell=True, cwd=dir , timeout=args.timeout, check=True, text=True
                           , stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            output("Main ran successfully for " + str(args.timeout) + " seconds", f)
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            output("Main terminated after " + str(duration) + " seconds", f)

            for line in e.stdout.split("\n"):
                if "error" in line.lower():
                    output("Error = " + line, f)
        # success block
        else:
            output("Main finished successfully under " + str(args.timeout) + " seconds", f)
        finally:
            shutil.rmtree(os.path.join(dir, ".test-venv"))
            output("----------------------------", f)

    elif os.path.isfile(main) and "pip/_internal" not in dir:
        output("Testing: " + dir, f)
        output("Folder has main but not requirements", f)
        output("----------------------------", f)
    elif os.path.isfile(requirements):
        output("Testing: " + dir, f)
        output("Folder has requirements but not main", f)
        output("----------------------------", f)

print("Starting test...")

if args.save:
    log_file = "test_" + datetime.datetime.now().strftime("%H:%M:%S") + ".txt"
    with open(log_file, "w") as f:
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
            test_directory(dirpath, f)
        else:
            test_directory(args.path, f)

        print("Test finished, results in: " + log_file)
else:
    if args.path is None:
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
            test_directory(dirpath)
    else:
        test_directory(args.path)

    print("Test finished")
