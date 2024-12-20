#!/usr/bin/env python3
import depthai as dai


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

    pipeline.start()
    remoteConnector.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = remoteConnector.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
