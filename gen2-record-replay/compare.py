#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
from libraries.depthai_replay import Replay
import skimage.metrics as m
#from skimage.measure import compare_ssim as ssim

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings/TEST720", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay object
replay = Replay(args.path)
# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.init_pipeline()
replay.disable_stream('color')

disp_out = pipeline.createXLinkOut()
disp_out.setStreamName("disp")
nodes.stereo.disparity.link(disp_out.input)

with dai.Device(pipeline) as device:
    replay.create_queues(device)
    dispQ = device.getOutputQueue(name="disp", maxSize=4, blocking=False)

    new = []
    old = []
    while replay.send_frames():
        old.append(replay.frames['disparity'])
        if len(old) > 4: old.pop(0)
        new.append(dispQ.get().getCvFrame())
        if len(new) > 4: new.pop(0)
        i = 0
        for o in old:
            for n in new:
                i+=1
                index = m.structural_similarity(o, n)
                print(i, index)

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')

