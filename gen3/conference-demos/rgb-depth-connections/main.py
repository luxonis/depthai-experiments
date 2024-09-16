import depthai as dai
import cv2

from host_bird_eye_view import BirdsEyeView
from host_rgb_conference_node import CombineOutputs
from host_display import Display

output_shape = (1280, 720)
nn_shape = (416, 416)

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    cam.initialControl.setManualFocus(130)
    cam_output = cam.requestOutput(output_shape)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(output_shape, dai.ImgFrame.Type.RGB888p),
        right=right.requestOutput(output_shape, dai.ImgFrame.Type.RGB888p)
    )
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)
    stereo.setOutputSize(*output_shape)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*nn_shape)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    manip.initialConfig.setKeepAspectRatio(False)
    manip.setMaxOutputFrameSize(nn_shape[0] * nn_shape[1] * 3)
    cam_output.link(manip.inputImage)

    nn_archive = dai.NNArchive(archive_path)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    spatialDetectionNetwork.setNNArchive(nn_archive)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(300)
    spatialDetectionNetwork.setDepthUpperThreshold(35000)
    spatialDetectionNetwork.input.setMaxSize(1)
    spatialDetectionNetwork.input.setBlocking(False)
    manip.out.link(spatialDetectionNetwork.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    spatialDetectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    spatialDetectionNetwork.setIouThreshold(0.5)

    bird_eye = pipeline.create(BirdsEyeView).build(spatialDetectionNetwork.out)

    combined = pipeline.create(CombineOutputs).build(
        color=cam_output,
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
