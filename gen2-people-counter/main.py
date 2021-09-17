#!/usr/bin/env python3
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np
import argparse
from time import monotonic
import itertools

from depthai_sdk import PipelineManager, NNetManager, PreviewManager
from depthai_sdk import crop_to_aspect_ratio

parentDir = Path(__file__).parent

#=====================================================================================
# To use a different NN, change `size` and `nnPath` here:
size = (544, 320)
nnPath = blobconverter.from_zoo("person-detection-retail-0013", shaves=8)
#=====================================================================================

# Labels
labelMap = ["background", "person"]

# Get argument first
parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn', type=str, help=".blob path")
parser.add_argument('-i', '--image', type=str,
                    help="Path to an image file to be used for inference (conflicts with -cam)")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI RGB camera for inference (conflicts with -vid)")
args = parser.parse_args()

# Whether we want to use images from host or rgb camera
IMAGE = not args.camera
nn_source = "host" if IMAGE else "color"

# Start defining a pipeline
pm = PipelineManager()
if not IMAGE:
    pm.create_color_cam(preview_size=size, xout=True)
    pv = PreviewManager(display=["color"], nn_source=nn_source)

nm = NNetManager(input_size=size, nn_family="mobilenet", labels=labelMap, confidence=0.5)
nn_pipeline = nm.create_nn_pipeline(pm.pipeline, pm.nodes, blob_path=nnPath, source=nn_source)
pm.set_nn_manager(nm)
pm.add_nn(nn_pipeline)

# Pipeline defined, now the device is connected to
with dai.Device(pm.pipeline) as device:
    nm.createQueues(device)
    if IMAGE:
        imgPaths = [args.image] if args.image else list(parentDir.glob('images/*.jpeg'))
        og_frames = itertools.cycle([crop_to_aspect_ratio(cv2.imread(str(imgPath)), size) for imgPath in imgPaths])
    else:
        pv.create_queues(device)

    while True:
        if IMAGE:
            frame = next(og_frames).copy()
            nm.sendInputFrame(frame)
        else:
            pv.prepare_frames(blocking=True)
            frame = pv.get("color")

        nn_data = nm.decode(nm.output_queue.get())
        nm.draw(frame, nn_data)
        cv2.putText(frame, f"People count: {len(nn_data)}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))
        cv2.imshow("color", frame)

        if cv2.waitKey(3000 if IMAGE else 1) == ord('q'):
            break
