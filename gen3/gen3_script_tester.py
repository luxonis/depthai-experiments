import os
from venv import EnvBuilder
import subprocess
import shutil
import datetime
import time

TIMEOUT = 30
SAVE_TO_FILE = True
PRINT = True

def output(text, f):
    if SAVE_TO_FILE:
        f.write(text+"\n")
    if PRINT:
        print(text)

def setup_venv_exe(dir, f):
    env_dir = os.path.join(dir, ".test-venv")
    env_builder = EnvBuilder(clear=True, with_pip=True, system_site_packages=False)
    env_builder.create(env_dir)
    env_bin = os.path.join(env_dir, 'bin')
    env_exe = os.path.join(env_bin, 'python3')

    try:
        pass
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

def test_directory(dir, f):
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
            subprocess.run(executable + " " + main, shell=True, cwd=dir , timeout=TIMEOUT, check=True, text=True
                           , stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            output("Main run successfully for " + str(TIMEOUT) + " seconds", f)
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            output("Main terminated after " + str(duration) + " seconds", f)

            for line in e.stdout.split("\n"):
                if "error" in line.lower():
                    output("Error = " + line, f)
        # success block
        else:
            output("Main finished successfully under " + str(TIMEOUT) + " seconds", f)
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

if SAVE_TO_FILE:
    log_file = "test_" + datetime.datetime.now().strftime("%H:%M:%S") + ".txt"
    with open(log_file, "w") as f:
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
            test_directory(dirpath, f)

        print("Test finished, results in: " + log_file)
else:
    for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
        test_directory(dirpath, None)

    print("Test finished")