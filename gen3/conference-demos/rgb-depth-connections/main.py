import depthai as dai
import cv2

from host_bird_eye_view import BirdsEyeView
from host_rgb_conference_node import CombineOutputs
from host_display import Display

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    cam.initialControl.setManualFocus(130)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput((1280, 720), dai.ImgFrame.Type.RGB888p),
        right=right.requestOutput((1280, 720), dai.ImgFrame.Type.RGB888p)
    )
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)
    stereo.setOutputSize(1280, 720)

    cap = dai.ImgFrameCapability()
    cap.type = dai.ImgFrame.Type.RGB888p
    cap.size.fixed((416, 416))
    cap.resizeMode = dai.ImgResizeMode.STRETCH

    nn_archive = dai.NNArchive(archive_path)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    spatialDetectionNetwork.setNNArchive(nn_archive)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(300)
    spatialDetectionNetwork.setDepthUpperThreshold(35000)
    cam.requestOutput(cap, False).link(spatialDetectionNetwork.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    spatialDetectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    spatialDetectionNetwork.setIouThreshold(0.5)

    bird_eye = pipeline.create(BirdsEyeView).build(spatialDetectionNetwork.out)

    combined = pipeline.create(CombineOutputs).build(
        color=cam.requestOutput((1280, 720), dai.ImgFrame.Type.NV12),
        depth=spatialDetectionNetwork.passthroughDepth,
        birdseye=bird_eye.output,
        detections=spatialDetectionNetwork.out
    )

    display = pipeline.create(Display).build(combined.output)
    display.setName("Luxonis")

    cv2.namedWindow("Luxonis", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Luxonis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Pipeline created.")
    pipeline.run()
