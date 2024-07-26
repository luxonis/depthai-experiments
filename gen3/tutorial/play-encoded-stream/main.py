#!/usr/bin/env python3

import depthai as dai
import subprocess as sp
from os import name as osName
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-res', '--resolution', default='4k',
                    choices={'1080', '4k', '12mp', '13mp'},
                    help="Select color camera resolution. Default: %(default)s")
parser.add_argument('-enc', '--encode', default='h264',
                    choices={'h264', 'h265', 'jpeg'},
                    help="Select encoding format. Default: %(default)s")
parser.add_argument('-fps', '--fps', type=float, default=30.0,
                    help="Camera FPS to configure. Default: %(default)s")
parser.add_argument('-v', '--verbose', default=False, action="store_true",
                    help="Prints latency for the encoded frame data to reach the app")
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


class PlayEncodedVideo(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    
    def build(self, enc_out : dai.Node.Output) -> "PlayEncodedVideo":
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        self.inputs['encoded_vid'].setBlocking(True)
        self.inputs['encoded_vid'].setMaxSize(30)
        return self
    

    def process(self, encoded_vid) -> None:
        try:
            data = encoded_vid.getData()
            if args.verbose:
                latms = (dai.Clock.now() - encoded_vid.getTimestamp()).total_seconds() * 1000
                # Writing to a different channel (stderr)
                # Also `ffplay` is printing things, adding a separator
                print(f'Latency: {latms:.3f} ms === ', file=sys.stderr)
            proc.stdin.write(data)
        except:
            self.stopPipeline()


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


with dai.Pipeline() as pipeline:
    # Define sources and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    videoEnc = pipeline.create(dai.node.VideoEncoder)

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(res_opts[args.resolution])
    camRgb.setFps(args.fps)
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), enc_opts[args.encode])

    # Linking
    camRgb.video.link(videoEnc.input)

    pipeline.create(PlayEncodedVideo).build(
        enc_out=videoEnc.bitstream
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")

    proc.stdin.close()
    print("process closed")
