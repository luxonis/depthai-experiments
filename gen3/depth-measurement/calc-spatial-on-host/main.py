#!/usr/bin/env python3

import cv2
import depthai as dai
from calc import HostSpatialsCalc
from host_depth_color_transform import DepthColorTransform
from host_display import Display
from keyboard_reader import KeyboardReader
from measure_distance import MeasureDistance


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

    measure_distance = pipeline.create(MeasureDistance).build(stereo.depth, device.readCalibration(), HostSpatialsCalc.INITIAL_ROI)

    calibdata = device.readCalibration()
    spatials = pipeline.create(HostSpatialsCalc).build(
        disparity_frames=depth_color_transform.output,
        measured_depth=measure_distance.output,
        keyboard_input=keyboard_reader.output
    )
    spatials.output_roi.link(measure_distance.roi_input)

    display = pipeline.create(Display).build(spatials.output)
    display.setName("Depth")
    display.setKeyboardInput(keyboard_reader.output)

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
