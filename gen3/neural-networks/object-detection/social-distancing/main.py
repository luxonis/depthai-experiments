import depthai as dai
import blobconverter
from host_social_distancing import SocialDistancing

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(544, 320)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    nn.setBlobPath(blobconverter.from_zoo(name="person-detection-retail-0013", shaves=5))
    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)
    nn.setBoundingBoxScaleFactor(0.5)
    nn.setDepthLowerThreshold(100)
    nn.setDepthUpperThreshold(5000)
    cam.preview.link(nn.input)
    stereo.depth.link(nn.inputDepth)

    social_distancing = pipeline.create(SocialDistancing).build(
        preview=cam.preview,
        nn=nn.out
    )
    social_distancing.inputs["preview"].setBlocking(False)
    social_distancing.inputs["preview"].setMaxSize(4)
    social_distancing.inputs["detections"].setBlocking(False)
    social_distancing.inputs["detections"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
