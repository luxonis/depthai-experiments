#!/usr/bin/env python3
import argparse
import cv2
import time
import depthai as dai
import blobconverter
import numpy as np
from libraries.depthai_replay import Replay
from utils.config import ConfigParser
from utils.draw import drawDets

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-lr', '--lrMode', default=False, action='store_true', help="Enable LeftRight check mode for stereo")
parser.add_argument('-sp', '--subpixelMode', default=False, action='store_true',  help="Enable subpixel mode for stereo")
parser.add_argument('-ext', '--extendedMode', default=False, action='store_true',  help="Enable extended disparity mode for stereo")
parser.add_argument('-rect', '--rectified', default=False, action='store_true', help="Show rectified left and right streams")
parser.add_argument('-s', '--scale', default=100, type=int, help="change the scale required to convert to meters")
parser.add_argument('-off', '--offset', default=320, type=int,
                    help="Offset to crop 1920x1200 -> 1280x1200 for depth, range 0..640. Default: %(default)s")
parser.add_argument('-nnb', '--nnBlob', default=None, type=str, help="Path to the NN model")
parser.add_argument('-nnc', '--nnConfig', default=None, type=str, help="Path to the NN config")
args = parser.parse_args()

scale = args.scale

# Create Replay object
replay = Replay(args.path, args.lrMode, args.subpixelMode, args.extendedMode, args.offset)
# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.init_pipeline()
# Resize color frames prior to sending them to the device
replay.set_resize_color((416, 416))

# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
replay.keep_aspect_ratio(True)

if (args.nnConfig is None) is (args.nnBlob is not None):
    raise ValueError("When specifying NN both config and model must be provided.")

use_nn = args.nnConfig is not None and args.nnBlob is not None

if use_nn:
    nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    nn.setBoundingBoxScaleFactor(0.4)
    nn.setDepthLowerThreshold(100)
    nn.setDepthUpperThreshold(500000)

    nn.setBlobPath(args.nnBlob)
    #nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)

    # parse config
    config_parser = ConfigParser(args.nnConfig)
    labels = config_parser.labels

    nn.setConfidenceThreshold(config_parser.confidence_threshold)
    nn.setNumClasses(config_parser.classes)
    nn.setCoordinateSize(config_parser.coordinates)
    nn.setAnchors(config_parser.anchors)
    nn.setAnchorMasks(config_parser.anchor_masks)
    nn.setIouThreshold(config_parser.iou_threshold)

    # # Link required inputs to the Spatial detection network
    nodes.color.out.link(nn.input)
    nodes.stereo.depth.link(nn.inputDepth)

    detOut = pipeline.create(dai.node.XLinkOut)
    detOut.setStreamName("det_out")
    nn.out.link(detOut.input)


dispOut = pipeline.create(dai.node.XLinkOut)
dispOut.setStreamName("disp_out")
nodes.stereo.disparity.link(dispOut.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
nodes.stereo.depth.link(depthOut.input)

right_s_out = pipeline.create(dai.node.XLinkOut)
right_s_out.setStreamName("rightS")
nodes.stereo.syncedRight.link(right_s_out.input)

left_s_out = pipeline.create(dai.node.XLinkOut)
left_s_out.setStreamName("leftS")
nodes.stereo.syncedLeft.link(left_s_out.input)

color_out = pipeline.create(dai.node.XLinkOut)
color_out.setStreamName("color_out")
nodes.color.out.link(color_out.input)

if args.rectified:
    rect_l_out = pipeline.create(dai.node.XLinkOut)
    rect_l_out.setStreamName("rectifiedLeft")
    nodes.stereo.rectifiedLeft.link(rect_l_out.input)

    rect_r_out = pipeline.create(dai.node.XLinkOut)
    rect_r_out.setStreamName("rectifiedRight")
    nodes.stereo.rectifiedRight.link(rect_r_out.input)

with dai.Device(pipeline) as device:
    replay.create_queues(device)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=1, blocking=False)
    dispQ = device.getOutputQueue(name="disp_out", maxSize=1, blocking=False)
    rightS_Q = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)
    leftS_Q = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)
    if args.rectified:
        rectL_Q = device.getOutputQueue(name="rectifiedLeft", maxSize=4, blocking=False)
        rectR_Q = device.getOutputQueue(name="rectifiedRight", maxSize=4, blocking=False)
    color_Q = device.getOutputQueue(name="color_out", maxSize=4, blocking=False)
    if use_nn:
        det_Q = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)


    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    color = (255, 0, 0)
    frames = {}
    dets = []

    # For RGB-depth blending
    rgbFrameName = "rightRect"
    depthFrameName = "disp" #"depth"
    blendDepthRatio = 50
    
    cv2.namedWindow("rgbd")
    def blendSliderCb(value):
        global blendDepthRatio
        blendDepthRatio = value
    cv2.createTrackbar("Blend depth %", "rgbd", blendDepthRatio, 100, blendSliderCb)

    cv2.namedWindow("disp")
    cv2.namedWindow("depth")
    points = None
    def cb(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONUP:
            if points == (x, y):
                points = None # Clear
            else:
                points = (x, y)
    cv2.setMouseCallback("disp", cb)
    cv2.setMouseCallback("depth", cb)
    cv2.setMouseCallback("rgbd", cb)

    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colormap[0] = [0, 0, 0]  # zero (invalidated) pixels as black

    paused = False
    stop = False

    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.send_frames():
        # rgbFrame = replay.lastFrame['color']x``
        # if mono:
        if color_Q.has():
            frames["color"] = color_Q.get().getCvFrame()
        if leftS_Q.has():
            frames["left"] = leftS_Q.get().getCvFrame()
        if rightS_Q.has():
            frames["right"] = rightS_Q.get().getCvFrame()
        if args.rectified:
            if rectL_Q.has():
                frames["leftRect"] = rectL_Q.get().getCvFrame()
            if rectR_Q.has():
                frames["rightRect"] = rectR_Q.get().getCvFrame()
        if dispQ.has():
            frames["disp"] = dispQ.get().getFrame()
        if depthQ.has():
            frames["depth"] = depthQ.get().getFrame()

        if use_nn:
            if frames.get("color") is not None:
                dets_obj = det_Q.get()
                drawDets(frames["color"], dets_obj.detections, labels)

            
        # This acts like a do-while loop, if not paused, we break from it.
        # It allows frame interaction (e.g depth measurement) also while paused
        while True:
            # Add an empty placeholder for RGB-D, to reuse the loop below
            if rgbFrameName in frames and depthFrameName in frames:
                frames["rgbd"] = np.empty([1])
            frameD = None
            for name, frame in frames.items():
                copy = frame.copy()
                if name == "disp":
                    copy = (copy*disparityMultiplier).astype(np.uint8)
                    # copy = cv2.normalize(copy, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    # copy = cv2.equalizeHist(copy)
                    copy = cv2.applyColorMap(copy, colormap)
                if name == "depth":
                    copy = cv2.normalize(copy, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    copy = cv2.equalizeHist(copy)
                    copy = cv2.applyColorMap(copy, colormap)
                if name == depthFrameName: frameD = copy
                if name == "rgbd":
                    frameRgb = frames[rgbFrameName]
                    # Need to have both frames in same format (BGR) before blending
                    if len(frameRgb.shape) < 3:
                        frameRgb = cv2.cvtColor(frameRgb, cv2.COLOR_GRAY2BGR).copy()
                    copy = cv2.addWeighted(frameD, blendDepthRatio/100, 
                                           frameRgb, 1 - blendDepthRatio/100, 0)
                if points is not None and (name in ["disp", "depth", rgbFrameName, "rgbd"]):
                    text = "{:.3f}m".format(frames["depth"][points[1]][points[0]] / scale)
                    cv2.circle(copy, points, 3, (255, 255, 255), -1)
                    cv2.circle(copy, points, 1, (0, 0, 0), -1)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                    disparity = frames["disp"][points[1]][points[0]]
                    text = "{:.3f}m H spec b".format(replay.disp_to_depth_factor_spec_baseline / disparity)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                    text = "{:.3f}m H calib b".format(replay.disp_to_depth_factor_calib_baseline / disparity)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                    text = "disp {}".format(disparity / replay.disp_levels)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 65), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)
                    cv2.putText(copy, text, (points[0] + 5, points[1] + 65), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

                # if name == "rgb":
                #     drawDets(copy)
                cv2.imshow(name, copy)

            key = cv2.waitKey(1)
            if key == ord('q'):
                stop = True
                break
            elif key == ord('.'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused: print('Replay paused. Press . to advance one frame, Space to unpause...')
            if not paused: break
            time.sleep(0.033)
        if stop: break
    if stop:
        print('Stopped')
    else:
        print('End of the recording')

