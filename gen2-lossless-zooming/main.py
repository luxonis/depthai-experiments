import depthai as dai
import cv2
import numpy as np
import time
import blobconverter

print(dai.__file__)

# Constants
SCENE_SIZE = (1920, 1080)

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
# Squash whole 4K frame into 300x300
cam.setPreviewKeepAspectRatio(False)
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)
cam.initialControl.setManualFocus(130)

# Create MobileNet detection network
mobilenet = pipeline.createMobileNetDetectionNetwork()
mobilenet.setBlobPath(str(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=5)))
mobilenet.setConfidenceThreshold(0.7)
cam.preview.link(mobilenet.input)

# Script node
script = pipeline.create(dai.node.Script)
mobilenet.out.link(script.inputs['dets'])
script.setScript("""
ORIGINAL_SIZE = (3840, 2160) # 4K
SCENE_SIZE = (1920, 1080) # 1080P
x_arr = []
y_arr = []
AVG_MAX_NUM=7
limits = [SCENE_SIZE[0] // 2, SCENE_SIZE[1] // 2] # xmin and ymin limits
limits.append(ORIGINAL_SIZE[0] - limits[0]) # xmax limit
limits.append(ORIGINAL_SIZE[1] - limits[1]) # ymax limit

cfg = ImageManipConfig()
size = Size2f(SCENE_SIZE[0], SCENE_SIZE[1])

def average_filter(x, y):
    x_arr.append(x)
    y_arr.append(y)
    if AVG_MAX_NUM < len(x_arr): x_arr.pop(0)
    if AVG_MAX_NUM < len(y_arr): y_arr.pop(0)
    x_avg = 0
    y_avg = 0
    for i in range(len(x_arr)):
        x_avg += x_arr[i]
        y_avg += y_arr[i]
    x_avg = x_avg / len(x_arr)
    y_avg = y_avg / len(y_arr)
    if x_avg < limits[0]: x_avg = limits[0]
    if y_avg < limits[1]: y_avg = limits[1]
    if limits[2] < x_avg: x_avg = limits[2]
    if limits[3] < y_avg: y_avg = limits[3]
    return x_avg, y_avg

while True:
    dets = node.io['dets'].get().detections
    if len(dets) == 0: continue

    coords = dets[0] # take first
    # Get detection center
    x = (coords.xmin + coords.xmax) / 2 * ORIGINAL_SIZE[0]
    y = (coords.ymin + coords.ymax) / 2 * ORIGINAL_SIZE[1] + 200

    x_avg, y_avg = average_filter(x,y)

    rect = RotatedRect()
    rect.size = size
    rect.center = Point2f(x_avg, y_avg)
    cfg.setCropRotatedRect(rect, False)
    node.io['cfg'].send(cfg)
""")
crop_manip = pipeline.create(dai.node.ImageManip)
crop_manip.setMaxOutputFrameSize(3110400)
crop_manip.initialConfig.setResize(1920, 1080)
script.outputs['cfg'].link(crop_manip.inputConfig)
cam.isp.link(crop_manip.inputImage)

# Link
xout = pipeline.createXLinkOut()
xout.setStreamName('1080P')
crop_manip.out.link(xout.input)

xoutFull = pipeline.createXLinkOut()
xoutFull.setStreamName('full')
cam.preview.link(xoutFull.input)

with dai.Device(pipeline) as device:
    qHq = device.getOutputQueue(name='1080P')
    qFull = device.getOutputQueue(name='full')
    # Main loop
    while True:
        if qHq.has():
            cv2.imshow('cropped', cv2.resize(qHq.get().getCvFrame(), (640,360)))
        if qFull.has():
            cv2.imshow('full', qFull.get().getCvFrame())
        # Update GUI and handle keypresses
        if cv2.waitKey(1) == ord('q'):
            break
