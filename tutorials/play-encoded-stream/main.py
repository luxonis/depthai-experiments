#!/usr/bin/env python3

import depthai as dai
import argparse
from os import name as osName
import subprocess as sp

from hostnodes.host_play_encoded_video import PlayEncodedVideo, WIDTH, HEIGHT
from hostnodes.host_display import Display

parser = argparse.ArgumentParser()
parser.add_argument(
    "-enc",
    "--encode",
    default="jpeg",
    choices={"h264", "h265", "jpeg"},
    help="Select encoding format. Default: %(default)s",
)
parser.add_argument(
    "-fps",
    "--fps",
    type=float,
    default=30.0,
    help="Camera FPS to configure. Default: %(default)s",
)
parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    action="store_true",
    help="Prints latency for the encoded frame data to reach the app",
)
args = parser.parse_args()


enc_opts = {
    "h264": dai.VideoEncoderProperties.Profile.H264_MAIN,
    "h265": dai.VideoEncoderProperties.Profile.H265_MAIN,
    "jpeg": dai.VideoEncoderProperties.Profile.MJPEG,
}

command = [
    "ffplay",
    "-i",
    "-",
    "-x",
    str(WIDTH),
    "-y",
    str(HEIGHT),
    "-framerate",
    "60",
    "-fflags",
    "nobuffer",
    "-flags",
    "low_delay",
    "-framedrop",
    "-strict",
    "experimental",
]

if osName == "nt":  # Running on Windows
    command = ["cmd", "/c"] + command

try:
    proc = sp.Popen(command, stdin=sp.PIPE)  # Start the ffplay process
except Exception as e:
    exit("Error: cannot run ffplay!\nTry running: sudo apt install ffmpeg")


with dai.Pipeline() as pipeline:
    # Define sources and output
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    video = camRgb.requestOutput(
        size=(WIDTH, HEIGHT), type=dai.ImgFrame.Type.NV12, fps=args.fps
    )

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(args.fps, enc_opts[args.encode])

    # Linking
    video.link(videoEnc.input)

    decoded = pipeline.create(PlayEncodedVideo).build(
        enc_out=videoEnc.bitstream, proc=proc
    )

    color = pipeline.create(Display).build(frame=decoded.output)
    color.setName("Color")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")

    proc.stdin.close()
    print("process closed")
