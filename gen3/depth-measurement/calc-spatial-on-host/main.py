#!/usr/bin/env python3

import depthai as dai
from calc import HostSpatialsCalc

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
    
    calibdata = device.readCalibration()
    host = pipeline.create(HostSpatialsCalc).build(
        stereo,
        calibdata
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
