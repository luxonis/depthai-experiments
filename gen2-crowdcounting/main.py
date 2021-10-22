#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import errno
import os
import blobconverter
from depthai_sdk import FPSHandler

'''
DM-Count crowd counting demo running on device with video input from host.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -v path/to/video

Model is taken from:
https://github.com/cvlab-stonybrook/DM-Count
and weights from:
https://github.com/tersekmatija/lwcc

DepthAI 2.9.0.0 is required. Blob was compiled using OpenVino 2021.4
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-v', '--video_path', help="Path to video frame", default="vids/virat.mp4")

args = parser.parse_args()

video_source = args.video_path

# resize input to smaller size for faster inference
NN_WIDTH = 960
NN_HEIGHT = 540


# --------------- Methods ---------------
# scale to [0 ... 255] and apply colormap
def decode_density_map(output_tensor):
    output = np.array(output_tensor) * 255
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_VIRIDIS)
    output_colors = cv2.resize(output_colors, (NN_WIDTH, NN_HEIGHT), interpolation = cv2.INTER_LINEAR)
    return output_colors

# merge 2 frames together
def show_output(output_colors, frame):
    return cv2.addWeighted(cv2.resize(frame, (output_colors.shape[1], output_colors.shape[0])), 0.6, output_colors, 0.4, 0)
    #return cv2.hconcat([frame, output_colors])

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1).flatten()


# --------------- Check input ---------------

if not args.camera and args.video_path is None:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")


if not args.camera:
    vid_path = Path(video_source)
    if not vid_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), video_source)

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Create Manip for image resizing and NN for count inference
manip = pipeline.createImageManip()
detection_nn = pipeline.createNeuralNetwork()

# Create output links, and in link for video
if args.camera:
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("rgb")
    cam_rgb.preview.link(cam_xout.input)
    cam_rgb.preview.link(detection_nn.input)
else:
    frame_xin = pipeline.createXLinkIn()
    frame_xin.setStreamName("in_nn")
    frame_xin.out.link(detection_nn.input)

nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("nn")


# setting node configs
nn_path = blobconverter.from_zoo(name = "dmcount_540x960", zoo_type = "depthai", shaves = 7)
detection_nn.setBlobPath(str(nn_path))
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(1)

# Linking
detection_nn.out.link(nn_xout.input)

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if args.camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
    else:
        cap = cv2.VideoCapture(str(Path(args.video_path).resolve().absolute()))
        detection_in = device.getInputQueue("in_nn")


    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=True)

    fps_handler = FPSHandler(maxTicks=2)


    def should_run():
        return cap.isOpened() if not args.camera else True

    def get_frame():
        if args.camera:
            in_rgb = q_rgb.get()
            new_frame = in_rgb.getCvFrame()
            return True, np.ascontiguousarray(new_frame)
        else:
            return cap.read()

    while should_run():

        read_correctly, frame = get_frame()

        if not read_correctly:
            break


        if not args.camera:
            img_data = dai.ImgFrame()
            img_data.setData(to_planar(frame, (NN_WIDTH, NN_HEIGHT)))
            img_data.setWidth(NN_WIDTH)
            img_data.setHeight(NN_HEIGHT)
            detection_in.send(img_data)

        fps_handler.tick("fps")

        in_nn = q_nn.get()

        # read output
        lay1 = np.array(in_nn.getFirstLayerFp16()).reshape((66,120))
        count = np.sum(lay1)    # predicted count is the sum of the density map

        # generate and append density map
        output_colors = decode_density_map(lay1)
        frame = show_output(output_colors, frame)

        # show fps and predicted count
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps_handler.tickFps("fps"))
        label_count = "Predicted count: {:.2f}".format(count)

        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        (w2, h2), _ = cv2.getTextSize(label_count, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)

        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.rectangle(frame, (0, 0), (w2 + 2, h2 + 6), color_white, -1)

        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)
        cv2.putText(frame, label_count, (2, 12), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        cv2.imshow("Predict count", frame)

        if cv2.waitKey(1) == ord('q'):
            break
