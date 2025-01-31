#!/usr/bin/env python3

import cv2
import depthai as dai
from calc import HostSpatialsCalc
from depthai_nodes.nodes import DepthColorTransform
from host_node.measure_distance import MeasureDistance
from host_node.host_display import Display
from host_node.keyboard_reader import KeyboardReader


device = dai.Device()
with dai.Pipeline(device) as pipeline:
    monoLeft = (
        pipeline.create(dai.node.Camera)
        .build(dai.CameraBoardSocket.CAM_B)
        .requestOutput((640, 480), type=dai.ImgFrame.Type.NV12)
    )
    monoRight = (
        pipeline.create(dai.node.Camera)
        .build(dai.CameraBoardSocket.CAM_C)
        .requestOutput((640, 480), type=dai.ImgFrame.Type.NV12)
    )

    stereo = pipeline.create(dai.node.StereoDepth).build(
        monoLeft, monoRight, presetMode=dai.node.StereoDepth.PresetMode.DEFAULT
    )

    depth_color_transform = pipeline.create(DepthColorTransform).build(stereo.disparity)
    depth_color_transform.setColormap(cv2.COLORMAP_JET)

    keyboard_reader = pipeline.create(KeyboardReader).build(
        depth_color_transform.out
    )

    measure_distance = pipeline.create(MeasureDistance).build(
        stereo.depth, device.readCalibration(), HostSpatialsCalc.INITIAL_ROI
    )

    calibdata = device.readCalibration()
    spatials = pipeline.create(HostSpatialsCalc).build(
        disparity_frames=depth_color_transform.out,
        measured_depth=measure_distance.output,
        keyboard_input=keyboard_reader.output,
    )
    spatials.output_roi.link(measure_distance.roi_input)

    display = pipeline.create(Display).build(spatials.output)
    display.setName("Depth")
    display.setKeyboardInput(keyboard_reader.output)

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
