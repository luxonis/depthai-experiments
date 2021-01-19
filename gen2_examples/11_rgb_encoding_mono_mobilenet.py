#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import subprocess

pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam.video.link(videoEncoder.input)

videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('h265')
videoEncoder.bitstream.link(videoOut.input)

cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/mobilenet-ssd.blob')).resolve().absolute()))

manip = pipeline.createImageManip()
manip.setResize(300, 300)
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
cam_left.out.link(manip.inputImage)
manip.out.link(detection_nn.input)

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")
cam_left.out.link(xout_left.input)

xout_manip = pipeline.createXLinkOut()
xout_manip.setStreamName("manip")
manip.out.link(xout_manip.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

device = dai.Device(pipeline)
device.startPipeline()

queue_size = 8
overwriteLRU = True #overwrite least recently used frame in queue if it gets full (not blocking)
q_left = device.getOutputQueue("left", queue_size, overwriteLRU)
q_manip = device.getOutputQueue("manip", queue_size, overwriteLRU)
q_nn = device.getOutputQueue("nn", queue_size, overwriteLRU)
q_rgb_enc = device.getOutputQueue('h265', maxSize=30, blocking=True)

frame = None
frame_manip = None
bboxes = []


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


videoFile = open('video.h265','wb')

while True:
    in_left = q_left.tryGet()
    in_manip = q_manip.tryGet()
    in_nn = q_nn.tryGet()
    in_rgb_enc = q_rgb_enc.tryGet()

    if in_rgb_enc is not None: 
        in_rgb_enc.getData().tofile(videoFile)

    if in_left is not None:
        shape = (in_left.getHeight(), in_left.getWidth())
        frame = in_left.getData().reshape(shape).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_manip is not None:
        shape = (3, in_manip.getHeight(), in_manip.getWidth())
        frame_manip = in_manip.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame_manip = np.ascontiguousarray(frame_manip)

    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

    if frame is not None:
        cv2.imshow("left", frame)

    if frame_manip is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame_manip, raw_bbox)
            cv2.rectangle(frame_manip, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("manip", frame_manip)

    if cv2.waitKey(1) == ord('q'):
        break

videoFile.close()

print("Converting stream file (.h265) into a video file (.mp4)...")
subprocess.check_call("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4".split())
print("Conversion successful, check video.mp4")
