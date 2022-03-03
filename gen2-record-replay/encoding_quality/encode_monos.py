#!/usr/bin/env python3
import argparse
import depthai as dai
from libraries.depthai_replay import Replay
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings/720NEW", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay object
replay = Replay(args.path)
# Initialize the pipeline. This will create required XLinkIn's and connect them together
replay.disable_stream('color')
pipeline, nodes = replay.init_pipeline()

def stream_out(name, size, fps, out):
    # Create XLinkOutputs for the stream
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(name)

    enc = pipeline.create(dai.node.VideoEncoder)
    enc.setDefaultProfilePreset(size, fps, dai.VideoEncoderProperties.Profile.MJPEG)
    # enc.setBitrateKbps(100000)
    # enc.setLossless(True)
    enc.setQuality(96)
    out.link(enc.input)
    enc.bitstream.link(xout.input)

stream_out("leftOut", (1280,720), 30, nodes.left.out)
stream_out("rightOut", (1280,720), 30, nodes.right.out)

with dai.Device(pipeline) as device:
    replay.create_queues(device)
    q = {}
    q['left'] = device.getOutputQueue(name="leftOut", maxSize=4, blocking=False)
    q['right'] = device.getOutputQueue(name="rightOut", maxSize=4, blocking=False)

    folder = Path(args.path) / 'q96'
    folder.mkdir(parents=False, exist_ok=False)
    files = {}
    for name in q:
        files[name] = open(str(folder / f"{name}.mjpeg"), 'wb')
    while replay.send_frames():
        for name in q:
            img = q[name].get().getCvFrame()
            files[name].write(img)
    for name in files:
        files[name].close()
    print('End of the recording')

