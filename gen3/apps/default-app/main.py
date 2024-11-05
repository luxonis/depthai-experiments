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
                "yolov6-nano",
                platform=platform.name,
                modelVersionSlug=f"r2-coco-{NN_DIMENSIONS[0]}x{NN_DIMENSIONS[1]}")
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
    
    outputToEncode = cameraNode.requestOutput((1440, 1080), type=dai.ImgFrame.Type.NV12)
    h264Encoder = pipeline.create(dai.node.VideoEncoder)
    h264Encoder.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H264_MAIN)
    outputToEncode.link(h264Encoder.input)

    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
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

    # Add the remote connector topics
    remoteConnector.addTopic("Raw video", outputToEncode)
    remoteConnector.addTopic("Video H264", h264Encoder.out)
    remoteConnector.addTopic("Detections", detectionNetwork.out)
    remoteConnector.addTopic("Depth", coloredDepth.output)

    remoteConnector.registerPipeline(pipeline)
    pipeline.run()
