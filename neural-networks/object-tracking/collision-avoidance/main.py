import depthai as dai

from collision_avoidance_node import CollisionAvoidanceNode
from fps_counter import FPSCounter

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform="RVC2", modelVersionSlug="r2-coco-512x288")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(512, 288)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(30)

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(monoLeft.out, monoRight.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    nn.setNNArchive(nn_archive)
    nn.setNumInferenceThreads(2)
    nn.setConfidenceThreshold(0.5)
    cam.preview.link(nn.input)
    stereo.depth.link(nn.inputDepth)

    fps_counter = pipeline.create(FPSCounter).build(cam.preview)

    collision_avoidance = pipeline.create(CollisionAvoidanceNode).build(cam.preview, nn.out)

    pipeline.run()