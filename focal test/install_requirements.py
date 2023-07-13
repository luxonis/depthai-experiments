import subprocess
import sys

in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_install = pip_call + ["install"]

if not in_venv:
    pip_install.append("--user")

subprocess.check_call([*pip_install, "pip", "-U"])
subprocess.check_call([*pip_install, "-r", "requirements.txt"])
try:
    subprocess.check_call([sys.executable, "-c", "import tkinter"])
except subprocess.CalledProcessError as ex:
    if sys.platform == "linux":
        print('Missing `python3-tk` on the system. Install with `sudo apt install python3-tk`')
    else:
        print('Missing python3-tkinter on the system.')