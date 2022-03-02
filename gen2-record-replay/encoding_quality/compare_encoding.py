#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
from libraries.depthai_replay import Replay
from skimage.measure import compare_ssim as ssim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings/720NEW", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay object
replay = Replay(args.path + '/q96')
# Initialize the pipeline. This will create required XLinkIn's and connect them together
replay.disable_stream('color')
pipeline, nodes = replay.init_pipeline()

nodes.stereo.initialConfig.setConfidenceThreshold(255)
nodes.stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
nodes.stereo.setLeftRightCheck(False)
nodes.stereo.setExtendedDisparity(False)
nodes.stereo.setSubpixel(False)

def stream_out(name, size, fps, out):
    # Create XLinkOutputs for the stream
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(name)
    out.link(xout.input)

stream_out("disparity", (1280,720), 30, nodes.stereo.disparity)

with dai.Device(pipeline) as device:
    replay.create_queues(device)
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    vc = cv2.VideoCapture(args.path + '/disparity.avi')
    vals = []
    while replay.send_frames():
        new = q.get().getCvFrame()
        ok, frame = vc.read()
        if not ok: break
        frame = frame[:,:,0]
        vals.append(ssim(frame, new))
    arr = np.array(vals)
    print('avrg',np.average(arr))

    print('End of the recording')

