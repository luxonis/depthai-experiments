#!/usr/bin/env python3
import depthai as dai
from rgb_conference_node import Display

modelDescription = dai.NNModelDescription(modelSlug="yolov6-nano", platform="RVC2")
archivePath = dai.getModelFromZoo(modelDescription)


with dai.Pipeline() as pipeline:

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.initialControl.setManualFocus(130)
    camRgb.setIspScale(2, 3) # Downscale color to match mono
    camRgb.setPreviewKeepAspectRatio(False)

    # Properties
    camRgb.setPreviewSize(512, 288) 
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=monoLeft.out, right=monoRight.out)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)

    nn_archive = dai.NNArchive(archivePath)
    spatialDetectionNetwork.setNNArchive(nn_archive)
    spatialDetectionNetwork.setConfidenceThreshold(0.3)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(300)
    spatialDetectionNetwork.setDepthUpperThreshold(35000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
    spatialDetectionNetwork.setIouThreshold(0.5)

    camRgb.preview.link(spatialDetectionNetwork.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    pipeline.create(Display).build(
        cam_rgb=camRgb.video,
        depth=spatialDetectionNetwork.passthroughDepth,
        nn_out=spatialDetectionNetwork.out
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
