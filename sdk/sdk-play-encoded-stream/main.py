import subprocess as sp
from os import name as os_name

from depthai_sdk import OakCamera

width, height = 720, 500
command = [
    "ffplay",
    "-i", "-",
    "-x", str(width),
    "-y", str(height),
    "-framerate", "60",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-framedrop",
    "-strict", "experimental"
]

if os_name == "nt":  # Running on Windows
    command = ["cmd", "/c"] + command

try:
    proc = sp.Popen(command, stdin=sp.PIPE)  # Start the ffplay process
except:
    exit("Error: cannot run ffplay!\nTry running: sudo apt install ffmpeg")

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', encode='h264', fps=60)
    oak.callback(color.out.encoded, lambda packet: proc.stdin.write(packet.frame))
    oak.start(blocking=True)

proc.stdin.close()
