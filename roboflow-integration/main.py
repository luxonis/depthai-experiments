import time
import json
import argparse

from pathlib import Path
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import depthai as dai
import numpy as np

from utils.roboflow import RoboflowUploader

BLOB_PATH = "models/mobilenet-ssd_openvino_2021.4_5shave.blob"
LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

def make_pipeline():
    # Pipeline
    pipeline = dai.Pipeline()

    # Camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(300, 300)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.setInterleaved(False)
    camRgb.setFps(60)

    # Detector
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # Image output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Detection output
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")

    # Link elements
    nn.passthrough.link(xoutRgb.input)
    camRgb.preview.link(nn.input)  # RGB buffer
    nn.out.link(nnOut.input)

    return pipeline


def parse_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", help="Roboflow API key")
    parser.add_argument("--dataset", help="Roboflow dataset ID")
    parser.add_argument(
        "--autoupload_threshold",
        help="Upload only the bounding boxex above certain threshold. Threshold of 0.5 set by default.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--autoupload_interval",
        help="Auto-upload images every `autoupload_interval` seconds. Infinite interval (no auto-upload) set by default.",
        default=float("inf"),
        type=float,
    )
    config = parser.parse_args()

    return config


def parse_dets(detections, confidence_thr=0.8):

    labels = [LABELS[d.label] for d in detections if d.confidence > confidence_thr]

    bboxes = [
        [300 * d.xmin, 300 * d.ymin, 300 * d.xmax, 300 * d.ymax]
        for d in detections
        if d.confidence > confidence_thr
    ]

    return labels, bboxes


# nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def overlay_boxes(frame, detections):

    # Overlay on a copy of image to keep the original
    frame = frame.copy()
    BLUE = (255, 0, 0)

    for detection in detections:
        bbox = frameNorm(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )
        cv2.putText(
            frame,
            LABELS[detection.label],
            (bbox[0] + 10, bbox[1] + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            BLUE,
        )
        cv2.putText(
            frame,
            f"{int(detection.confidence * 100)}%",
            (bbox[0] + 10, bbox[1] + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            BLUE,
        )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BLUE, 2)

    return frame


def upload_all(uploader, frame: np.ndarray, labels: list, bboxes: list, fname: str):
    """
    Uploads `frame` as an image to Roboflow and saves it under `fname`.jpg
    Then, upload annotations  with corresponding `bboxes` and `frame`
    """

    # Upload image frame. Retreive Roboflow's image_id
    img_id = uploader.upload_image(frame, fname)

    # Annotate the image we just uploaded
    uploader.upload_annotation(img_id, fname=fname, labels=labels, bboxes=bboxes)


def get_last_synced_pair(rgb_deque, dets_deque):
    # Returns (frame, dets) with the highest available seq_n or (None, None) if no mach found.

    # rgb_deque sorted by seq_n
    rgb_deque_s = sorted(rgb_deque, key=lambda x: x[1], reverse=True)

    # Dict mapping seq_n: dets
    seq2dets = {seq_n: det for (det, seq_n) in dets_deque}

    # ODict mapping {seq_n: (frame, dets)}. Ignores seq_n without dets
    seq2frames_dets = OrderedDict(
        (
            (seq_n, (frame, seq2dets.get(seq_n)))
            for frame, seq_n in rgb_deque_s
            if seq2dets.get(seq_n) is not None
        )
    )

    # Return matches if any exist
    if len(seq2frames_dets) > 0:
        frame, dets = list(seq2frames_dets.values())[0]
    else:
        frame, dets = None, None

    return frame, dets


if __name__ == "__main__":

    # Parse config
    config = parse_cmd_args()

    UPLOAD_THR = config.autoupload_threshold
    DATASET = config.dataset
    API_KEY = config.api_key

    # Initialize variables
    frame = None
    detections = []
    last_upload_ts = time.monotonic()
    WHITE = (255, 255, 255)

    # Deques for detections and frames. Used for syncing frame<->detections pairs.
    rgb_deque = deque(maxlen=10)
    det_deque = deque(maxlen=10)

    # Wrapper around Roboflow upload/annotate API
    uploader = RoboflowUploader(dataset_name=DATASET, api_key=API_KEY)

    # Executor to handle uploads asynchronously
    # For real-time uploads at ~10Hz we spawn 40 threads
    executor = ThreadPoolExecutor(max_workers=40)

    # DAI pipeline
    pipeline = make_pipeline()

    with dai.Device(pipeline) as device:

        queue_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        queue_dets = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        print("Press 'enter' to upload annotated image to Roboflow. Press 'q' to exit.")

        cnt = 0 
        while True:

            cnt += 1

            rgb_msg = queue_rgb.get()  # instance of depthai.ImgFrame
            det_msg = queue_dets.get()  # instance of depthai.ImgDetections.getSequenceNum

            # Obtain sequence numbers to sync frames
            rgb_seq = rgb_msg.getSequenceNum()
            det_seq = det_msg.getSequenceNum()

            print(f"Inter {cnt}  RGB seq: {rgb_seq} Det seq {det_seq}")

            # Get frame and dets
            frame = rgb_msg.getCvFrame()  # np.ndarray / BGR CV Mat
            dets = det_msg.detections  # list of depthai.ImgDetection

            # Put (object, seq_n) tuples in a queue
            rgb_deque.append((frame, rgb_seq))
            det_deque.append((dets, det_seq))

            frame, dets = get_last_synced_pair(rgb_deque, det_deque)

            # Display results
            frame_with_boxes = overlay_boxes(frame, dets)
            cv2.imshow("Roboflow Demo", frame_with_boxes)

            # Time from last upload in seconds
            dt = time.monotonic() - last_upload_ts

            # Handle user input
            key = cv2.waitKey(1)

            if key == ord("q"):
                # q pressed
                exit()
            elif key == 13:
                # If enter is pressed, upload all dets without thresholding
                labels, bboxes = parse_dets(dets, confidence_thr=0.0)
                print("INFO: Enter pressed. Uploading grabbed frame!")
                executor.submit(
                    upload_all, uploader, frame, labels, bboxes, int(1000 * time.time())
                )
            elif dt > config.autoupload_interval:
                # Auto-upload annotations with confidence above UPLOAD_THR every `autoupload_interval` seconds
                labels, bboxes = parse_dets(dets, confidence_thr=UPLOAD_THR)

                if len(bboxes) > 0:
                    print(
                        f"INFO: Auto-uploading grabbed frame with {len(bboxes)} annotations!"
                    )
                    executor.submit(
                        upload_all,
                        uploader,
                        frame,
                        labels,
                        bboxes,
                        int(1000 * time.time()),
                    )
                    last_upload_ts = time.monotonic()
                else:
                    pass
                    # No detections. Could add a debug message here:
                    # print(f"DEBUG: No detections with confidence above {UPLOAD_THR}. Not uploading!")
