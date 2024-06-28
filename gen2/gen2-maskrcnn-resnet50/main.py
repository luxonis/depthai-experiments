#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np
import argparse

from depthai_sdk import PipelineManager, NNetManager, PreviewManager, Previews, FPSHandler, toTensorResult, frameNorm

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', help="Threshold for filtering out detections with lower probability", default=0.5, type=float)
parser.add_argument('-rt', '--region_threshold', help="Threshold for filtering out mask points with low probability", default=0.5, type=float)


args = parser.parse_args()

NN_SHAPE = 300, 300
NN_PATH = str(blobconverter.from_zoo("mask_rcnn_resnet50_coco_300x300", shaves=6, zoo_type="depthai"))
COLORS = np.random.random(size=(256, 3)) * 256
THRESHOLD = args.threshold
REGION_THRESHOLD = args.region_threshold
LABEL_MAP = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",          "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "street sign", "stop sign",    "parking meter",
    "bench",          "bird",       "cat",           "dog",           "horse",       "sheep",        "cow",
    "elephant",       "bear",       "zebra",         "giraffe",       "hat",         "backpack",     "umbrella",
    "shoe",           "eye glasses","handbag",       "tie",           "suitcase",    "frisbee",      "skis",
    "snowboard",      "sports ball","kite",          "baseball bat",  "baseball glove","skateboard", "surfboard",
    "tennis racket",  "bottle",     "plate",         "wine glass",    "cup",         "fork",         "knife",
    "spoon",          "bowl",       "banana",        "apple",         "sandwich",    "orange",       "broccoli",
    "carrot",         "hot dog",    "pizza",         "donut",         "cake",        "chair",        "sofa",
    "pottedplant",    "bed",        "mirror",        "diningtable",   "window",      "desk",         "toilet",
    "door",           "tvmonitor",  "laptop",        "mouse",         "remote",      "keyboard",     "cell phone",
    "microwave",      "oven",       "toaster",       "sink",          "refrigerator","blender",      "book",
    "clock",          "vase",       "scissors",      "teddy bear",    "hair drier",  "toothbrush"
]

# Start defining a pipeline
pm = PipelineManager()
pm.createColorCam(previewSize=NN_SHAPE, fullFov=False, xout=False)
pm.nodes.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

nm = NNetManager(inputSize=NN_SHAPE)
pm.setNnManager(nm)
pm.addNn(
    nm.createNN(pm.pipeline, pm.nodes, NN_PATH),
    xoutNnInput=True)
fps = FPSHandler()
pv = PreviewManager(display=[Previews.nnInput.name], fpsHandler=fps)

def show_boxes_and_regions(frame, boxes, masks):
    for i, box in enumerate(boxes):
        if box[0] == -1:
            break

        cls = int(box[1])
        prob = box[2]

        if prob < THRESHOLD:
            continue

        bbox = frameNorm(frame, box[-4:])
        cv2.rectangle(frame, (bbox[0], bbox[1]-15), (bbox[2], bbox[1]), COLORS[cls], -1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[cls], 1)
        cv2.putText(frame, f"{LABEL_MAP[cls-1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 2)
        cv2.putText(frame, f"{LABEL_MAP[cls-1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)

        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        mask = cv2.resize(masks[i, cls], (bbox_w, bbox_h))
        mask = mask > REGION_THRESHOLD

        roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        roi[mask] = roi[mask] * 0.6 + COLORS[cls] * 0.4
        frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi


# Pipeline is defined, now we can connect to the device
with dai.Device(pm.pipeline) as device:
    nm.createQueues(device)
    pv.createQueues(device)

    while True:
        fps.tick('color')
        pv.prepareFrames(blocking=True)
        frame = pv.get(Previews.nnInput.name)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = toTensorResult(nm.outputQueue.get())
        boxes = output["DetectionOutput_647"].squeeze()
        masks = output["Sigmoid_733"]

        show_boxes_and_regions(frame, boxes, masks)
        fps.drawFps(frame, Previews.nnInput.name)
        cv2.imshow(Previews.nnInput.name, frame)

        if cv2.waitKey(1) == ord('q'):
            break