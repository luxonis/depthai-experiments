import argparse
import time
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import blobconverter
import cv2
import depthai as dai
import numpy as np

from utils.roboflow import RoboflowUploader

BLOB_PATH = blobconverter.from_zoo(name="mobilenet-ssd", shaves=5)
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
    camRgb.setInterleaved(False)
    camRgb.setFps(30)
    camRgb.setPreviewKeepAspectRatio(False)

    # Detector
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # 300x300 RGB image output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    xoutRgb.input.setBlocking(False)
    xoutRgb.input.setQueueSize(1)

    # Detection output
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")

    # Hi-Res NV12 output
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutVideo.setStreamName("nv12")
    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    # Link elements
    nn.passthrough.link(xoutRgb.input)
    camRgb.preview.link(nn.input)  # RGB buffer
    nn.out.link(nnOut.input)
    camRgb.video.link(xoutVideo.input)  # NV12 to xoutVideo

    return pipeline


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", help="Roboflow API key")
    parser.add_argument("--dataset", help="Roboflow dataset ID")
    parser.add_argument(
        "--hires_uploads",
        help="Upload a larger-size image to roboflow instead of the 300x300 NN input image",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--autoupload_threshold",
        help="Upload only the bounding boxex above certain threshold. Threshold of 0.5 set by default.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--autoupload_interval",
        help="Auto-upload assets every `autoupload_interval` seconds. Infinite interval (no auto-upload) set by default.",
        default=float("inf"),
        type=float,
    )
    parser.add_argument(
        "--upload_res",
        help="Resolution of uploaded assets in WxH format. 300x300 by default.",
        default="300x300",
        type=str,
    )
    config = parser.parse_args()

    return config


def parse_dets(detections, image_shape, confidence_thr=0.8):
    X, Y = image_shape
    labels = [LABELS[d.label] for d in detections if d.confidence > confidence_thr]

    bboxes = [
        [X * d.xmin, Y * d.ymin, X * d.xmax, Y * d.ymax]
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
    H, W, C = frame.shape
    uploader.upload_annotation(
        img_id, fname=fname, labels=labels, bboxes=bboxes, img_w=W, img_h=H
    )


def get_last_synced_pair(img_deque, dets_deque):
    # Returns (frame, dets) with the highest available seq_n or (None, None) if no mach found.

    # img_deque sorted by seq_n
    img_deque_s = sorted(img_deque, key=lambda x: x[1], reverse=True)

    # Dict mapping seq_n: dets
    seq2dets = {seq_n: det for (det, seq_n) in dets_deque}

    # ODict mapping {seq_n: (frame, dets)}. Ignores seq_n without dets
    seq2frames_dets = OrderedDict(
        (
            (seq_n, (frame, seq2dets.get(seq_n)))
            for frame, seq_n in img_deque_s
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

    # Constant params
    DEV_QUEUE_SIZE = 5
    SYNC_QUEUE_SIZE = 20  # Sync queue should be larger than device queue
    UPLOAD_THR = config.autoupload_threshold
    DATASET = config.dataset
    API_KEY = config.api_key

    # Parse resolution '300x300' -> 300, 300
    W, H = map(int, config.upload_res.split("x"))

    # Initialize variables
    frame = None
    detections = []
    last_upload_ts = time.monotonic()
    WHITE = (255, 255, 255)

    # Deques for detections and frames. Used for syncing frame<->detections pairs.
    img_deque = deque(maxlen=SYNC_QUEUE_SIZE)
    det_deque = deque(maxlen=SYNC_QUEUE_SIZE)

    # Wrapper around Roboflow upload/annotate API
    uploader = RoboflowUploader(dataset_name=DATASET, api_key=API_KEY)

    # Executor to handle uploads asynchronously
    # For real-time uploads at ~10Hz we spawn 40 threads
    executor = ThreadPoolExecutor(max_workers=40)

    # DAI pipeline
    pipeline = make_pipeline()

    with dai.Device(pipeline) as device:

        # For 300x300 resolution -> use the RGB input
        # For any other resolution -> use NV12 input
        queue_frames = device.getOutputQueue(
            name="rgb" if config.upload_res == "300x300" else "nv12",
            maxSize=DEV_QUEUE_SIZE,
            blocking=False,
        )
        queue_dets = device.getOutputQueue(
            name="nn", maxSize=DEV_QUEUE_SIZE, blocking=False
        )

        print("Press 'enter' to upload annotated image to Roboflow. Press 'q' to exit.")

        while True:

            img_msg = queue_frames.get()  # instance of depthai.ImgFrame
            det_msg = queue_dets.get()  # instance of depthai.ImgDetections

            # Obtain sequence numbers to sync frames
            img_seq = img_msg.getSequenceNum()
            det_seq = det_msg.getSequenceNum()

            # Get frame and dets
            frame = img_msg.getCvFrame()  # np.ndarray / BGR CV Mat
            dets = det_msg.detections  # list of depthai.ImgDetection

            # Put (object, seq_n) tuples in a queue
            img_deque.append((frame, img_seq))
            det_deque.append((dets, det_seq))

            # Get synced pairs
            frame, dets = get_last_synced_pair(img_deque, det_deque)

            if frame is None or dets is None:
                continue

            # Resize to desired upload resolution
            frame = cv2.resize(frame, (W, H))

            # Display results
            frame_with_boxes = overlay_boxes(frame, dets)
            cv2.imshow("Roboflow Demo", frame_with_boxes)

            # Time from last upload in seconds
            dt = time.monotonic() - last_upload_ts

            # Handle user input
            key = cv2.waitKey(1)

            # Decide which frame to upload and obtain its shape (H, W, C)

            if key == ord("q"):
                # q pressed
                exit()
            elif key == 13:
                # If enter is pressed, upload all dets without thresholding
                labels, bboxes = parse_dets(dets, (W, H), confidence_thr=0.0)
                print("INFO: Enter pressed. Uploading grabbed frame!")
                executor.submit(
                    upload_all,
                    uploader,
                    frame,
                    labels,
                    bboxes,
                    int(1000 * time.time()),
                )
            elif dt > config.autoupload_interval:
                # Auto-upload annotations with confidence above UPLOAD_THR every `autoupload_interval` seconds
                labels, bboxes = parse_dets(dets, (W, H), confidence_thr=UPLOAD_THR)

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
                    print(
                        f"INFO: Auto-uploading grabbed frame with {len(bboxes)} annotations!"
                    )
                    # No detections. Could add a debug message here:
                    # print(f"DEBUG: No detections with confidence above {UPLOAD_THR}. Not uploading!")
