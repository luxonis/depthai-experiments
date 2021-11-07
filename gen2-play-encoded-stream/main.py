#!/usr/bin/env python3

import depthai as dai
import subprocess as sp
from os import name as osName

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
xout = pipeline.createXLinkOut()

xout.setStreamName("h264")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
videoEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

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

if osName == "nt":  # Running on Windows
    command = ["cmd", "/c"] + command

try:
    proc = sp.Popen(command, stdin=sp.PIPE)  # Start the ffplay process
except:
    exit("Error: cannot run ffplay!\nTry running: sudo apt install ffmpeg")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h264", maxSize=30, blocking=True)

    try:
        while True:
            data = q.get().getData()  # Blocking call, will wait until new data has arrived
            proc.stdin.write(data)
    except:
        pass

    proc.stdin.close()
