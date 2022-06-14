#!/usr/bin/env python3

import depthai as dai
import subprocess as sp
from os import name as osName
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-res', '--resolution', default='4k',
                    choices={'1080', '4k', '12mp', '13mp'},
                    help="Select color camera resolution. Default: %(default)s")
parser.add_argument('-enc', '--encode', default='h264',
                    choices={'h264', 'h265', 'jpeg'},
                    help="Select encoding format. Default: %(default)s")
parser.add_argument('-fps', '--fps', type=float, default=30.0,
                    help="Camera FPS to configure. Default: %(default)s")
args = parser.parse_args()

res_opts = {
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '4k':   dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '13mp': dai.ColorCameraProperties.SensorResolution.THE_13_MP,
}

enc_opts = {
    'h264': dai.VideoEncoderProperties.Profile.H264_MAIN,
    'h265': dai.VideoEncoderProperties.Profile.H265_MAIN,
    'jpeg': dai.VideoEncoderProperties.Profile.MJPEG,
}

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
videoEnc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("enc")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(res_opts[args.resolution])
camRgb.setFps(args.fps)
videoEnc.setDefaultProfilePreset(camRgb.getFps(), enc_opts[args.encode])

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
    q = device.getOutputQueue(name="enc", maxSize=30, blocking=True)

    try:
        while True:
            data = q.get().getData()  # Blocking call, will wait until new data has arrived
            proc.stdin.write(data)
    except:
        pass

    proc.stdin.close()
