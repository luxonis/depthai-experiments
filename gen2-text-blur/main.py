#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
from utils.utils import get_boxes

'''
Text blurring demo running on device, text detection is from:
https://github.com/MhLiao/DB


Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Onnx for text detection is taken from
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/145_text_detection_db,
and exported with scaling and mean_values flag.
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/text_detection_db_480x640_openvino_2021.4_6shave.blob', type=str)
parser.add_argument("-bt", "--box_thresh", help="set the confidence threshold of boxes", default=0.2, type=float)
parser.add_argument("-t", "--thresh", help="set the bitmap threshold", default=0.01, type=float)
parser.add_argument("-ms", "--min_size", default=1, type=int, help='set min size of box')
parser.add_argument("-mc", "--max_candidates", default=75, type=int, help='maximum number of candidate boxes')


args = parser.parse_args()

nn_path = args.nn_model
MAX_CANDIDATES = args.max_candidates
MIN_SIZE = args.min_size
BOX_THRESH = args.box_thresh
THRESH = args.thresh

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 640, 480


# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)


# Define a neural network that will detect text
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Define camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(NN_WIDTH, NN_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create outputs
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")


cam.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_cam.input)
detection_nn.out.link(xout_nn.input)


# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()

        frame = in_frame.getCvFrame()

        # Get output layer
        pred = np.array(in_nn.getLayerFp16("out")).reshape((480, 640))

        # Show output mask
        cv2.imshow("Preds", (pred * 255).astype(np.uint8))

        # Decode
        boxes, scores = get_boxes(pred, THRESH, BOX_THRESH, MIN_SIZE, MAX_CANDIDATES)

        # Blur image
        blur = cv2.GaussianBlur(frame, (49, 49), 30)

        for i, box in enumerate(boxes):

            # Draw boxes
            #cv2.rectangle(frame, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), (255, 0, 0), 1)
            #cv2.putText(frame, f"Score: {scores[i]:.2f}", (box[0,0], box[0,1]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0))

            # Blur boxes
            x1, y1, x2, y2 = box[0, 0] - 5, box[0, 1] - 5, box[2, 0] + 5, box[2, 1] + 5
            x1, x2 = np.clip([x1, x2], 0, frame.shape[1])
            y1, y2 = np.clip([y1, y2], 0, frame.shape[0])
            frame[y1:y2, x1:x2] = blur[y1:y2, x1:x2]

        # Show FPS
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # Show frame
        cv2.imshow("Detections", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break