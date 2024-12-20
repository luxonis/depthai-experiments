#!/usr/bin/env python3
import cv2
import depthai as dai
from host_node.host_depth_color_transform import DepthColorTransform


STEREO_RESOLUTION = (800, 600)
NN_DIMENSIONS = (512,288)

remoteConnector = dai.RemoteConnection()

device = dai.Device()
platform = device.getPlatform()
device.setIrLaserDotProjectorIntensity(1)
with dai.Pipeline(device) as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    nnArchive = dai.NNArchive(
        dai.getModelFromZoo(
            dai.NNModelDescription(
                model=f"yolov6-nano:r2-coco-{NN_DIMENSIONS[0]}x{NN_DIMENSIONS[1]}",
                platform=platform.name)
        )
    )
    if platform == dai.Platform.RVC2:
        detectionNetwork = pipeline.create(dai.node.DetectionNetwork)
        cameraNode.requestOutput(NN_DIMENSIONS, dai.ImgFrame.Type.BGR888p).link(detectionNetwork.input)
        detectionNetwork.setNNArchive(nnArchive, numShaves=6)
    else:
        detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(
            cameraNode.requestOutput(NN_DIMENSIONS, dai.ImgFrame.Type.BGR888i),
            nnArchive
        )

    ENCODING = dai.VideoEncoderProperties.Profile.MJPEG if platform == dai.Platform.RVC2 else dai.VideoEncoderProperties.Profile.H264_MAIN
    outputToEncode = cameraNode.requestOutput((1440, 1080), type=dai.ImgFrame.Type.NV12)
    colorEncoder = pipeline.create(dai.node.VideoEncoder)
    colorEncoder.setDefaultProfilePreset(30, ENCODING)
    outputToEncode.link(colorEncoder.input)

    # Add the remote connector topics
    remoteConnector.addTopic("Encoded Video", colorEncoder.out)
    remoteConnector.addTopic("Detections", detectionNetwork.out)

    # Stereo depth - only for stereo devices
    cameraFeatures = device.getConnectedCameraFeatures()

    cam_mono_1: dai.CameraBoardSocket | None = None
    cam_mono_2: dai.CameraBoardSocket | None = None
    for feature in cameraFeatures:
        if dai.CameraSensorType.MONO in feature.supportedTypes:
            if cam_mono_1 is None:
                cam_mono_1 = feature.socket
            else:
                cam_mono_2 = feature.socket
                break
    if cam_mono_1 and cam_mono_2:
        left_cam = pipeline.create(dai.node.Camera).build(cam_mono_1)
        right_cam = pipeline.create(dai.node.Camera).build(cam_mono_2)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_cam.requestFullResolutionOutput(dai.ImgFrame.Type.NV12),
            right=right_cam.requestFullResolutionOutput(dai.ImgFrame.Type.NV12),
            presetMode=dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        if platform == dai.Platform.RVC2:
            stereo.setOutputSize(*STEREO_RESOLUTION)
        
        coloredDepth = pipeline.create(DepthColorTransform).build(stereo.disparity)
        coloredDepth.setColormap(cv2.COLORMAP_JET)
        
        coloredDepthManip = pipeline.create(dai.node.ImageManipV2)
        coloredDepthManip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        if platform != dai.Platform.RVC2:
            coloredDepthManip.initialConfig.setOutputSize(480, 640)
        coloredDepth.output.link(coloredDepthManip.inputImage)

        depthEncoder = pipeline.create(dai.node.VideoEncoder)
        depthEncoder.setDefaultProfilePreset(30, ENCODING)
        coloredDepthManip.out.link(depthEncoder.input) 

        remoteConnector.addTopic("Encoded Depth", depthEncoder.out)

    pipeline.start()
    remoteConnector.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = remoteConnector.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
