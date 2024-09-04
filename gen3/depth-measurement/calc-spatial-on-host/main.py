#!/usr/bin/env python3

import cv2
import depthai as dai
from calc import HostSpatialsCalc
from host_depth_color_transform import DepthColorTransform
from host_display import Display
from keyboard_reader import KeyboardReader

device = dai.Device()
with dai.Pipeline(device) as pipeline:

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  
    
    stereo = pipeline.create(dai.node.StereoDepth).build(monoLeft.out, monoRight.out)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    depth_color_transform = pipeline.create(DepthColorTransform).build(stereo.disparity, stereo.initialConfig.getMaxDisparity())
    depth_color_transform.setColormap(cv2.COLORMAP_JET)

    keyboard_reader = pipeline.create(KeyboardReader).build(depth_color_transform.output)
    
    calibdata = device.readCalibration()
    spatials = pipeline.create(HostSpatialsCalc).build(
        depth_color_transform.output,
        stereo.depth,
        calibdata,
        keyboard_reader.output
    )

    display = pipeline.create(Display).build(spatials.output, keyboard_reader.output)
    display.setName("Depth")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
