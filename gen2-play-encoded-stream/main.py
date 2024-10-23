#!/usr/bin/env python3

import depthai as dai
import subprocess as sp
from os import name as osName
import argparse
import sys
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', default='rgb',
                    choices={'rgb', 'left', 'right', 'cama', 'camb', 'camc', 'camd', 'came', 'camf'},
                    help="Select encoding format. Default: %(default)s")
parser.add_argument('-res', '--resolution', default='4k',
                    choices={'1080', '1200', '1500', '1520', '1560', '3000', '4k', '12mp', '13mp'},
                    help="Select color camera resolution. Default: %(default)s")
parser.add_argument('-enc', '--encode', default='h264',
                    choices={'h264', 'h265', 'jpeg', 'none'},
                    help="Select encoding format. Default: %(default)s")
parser.add_argument('-fps', '--fps', type=float, default=30.0,
                    help="Camera FPS to configure. Default: %(default)s")
parser.add_argument('-v', '--verbose', default=False, action="store_true",
                    help="Prints latency for the encoded frame data to reach the app")
args = parser.parse_args()

cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.RGB,
    'left' : dai.CameraBoardSocket.LEFT,
    'right': dai.CameraBoardSocket.RIGHT,
    'cama' : dai.CameraBoardSocket.CAM_A,
    'camb' : dai.CameraBoardSocket.CAM_B,
    'camc' : dai.CameraBoardSocket.CAM_C,
    'camd' : dai.CameraBoardSocket.CAM_D,
    'came' : dai.CameraBoardSocket.CAM_E,
    'camf' : dai.CameraBoardSocket.CAM_F,
}

res_opts = {
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '1200': dai.ColorCameraProperties.SensorResolution.THE_1200_P,
#    '1500': dai.ColorCameraProperties.SensorResolution.THE_2000X1500,
#    '1520': dai.ColorCameraProperties.SensorResolution.THE_2028X1520,
#    '1560': dai.ColorCameraProperties.SensorResolution.THE_2104X1560,
    '3000': dai.ColorCameraProperties.SensorResolution.THE_4000X3000,
    '4k':   dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '13mp': dai.ColorCameraProperties.SensorResolution.THE_13_MP,
}

enc_opts = {
    'h264': dai.VideoEncoderProperties.Profile.H264_BASELINE,
    'h265': dai.VideoEncoderProperties.Profile.H265_MAIN,
    'jpeg': dai.VideoEncoderProperties.Profile.MJPEG,
}

filename_opts = {
    'h264': 'video.h264',
    'h265': 'video.h265',
    'jpeg': 'video.mjpeg',
}

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
if args.encode != 'none': videoEnc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("enc")

# Properties
camRgb.setBoardSocket(cam_socket_opts[args.camera])
camRgb.setResolution(res_opts[args.resolution])
camRgb.setFps(args.fps)
if args.encode != 'none': 
    videoEnc.setDefaultProfilePreset(camRgb.getFps() + 1, enc_opts[args.encode])
    videoEnc.setBitrateKbps(100000)
    #videoEnc.setMisc("max-bframes", 1)

# Linking
if args.encode != 'none':
    camRgb.isp.link(videoEnc.input)
    videoEnc.bitstream.link(xout.input)
else:
    camRgb.isp.link(xout.input)

width, height = 720, 500
command = [
    "ffplay",
    "-i", "-",
    "-x", str(width),
    "-y", str(height),
    "-hide_banner",
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
f = open(filename_opts[args.encode], 'wb')
with dai.Device(pipeline) as device:
    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="enc", maxSize=30, blocking=True)

    try:
        while True:
            pkt = q.get()  # Blocking call, will wait until new data has arrived
            if args.verbose:
                latms = (dai.Clock.now() - pkt.getTimestamp()).total_seconds() * 1000
                # Writing to a different channel (stderr)
                # Also `ffplay` is printing things, adding a separator
                print(f'Latency: {latms:.3f} ms === ', file=sys.stderr)
            if args.encode == 'none':
                cv2.imshow('cam', pkt.getCvFrame())
                cv2.waitKey(1)
            else:
                data = pkt.getData()
                data.tofile(f)
                proc.stdin.write(data)
    except:
        pass

    proc.stdin.close()
    f.close()
