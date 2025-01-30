import depthai as dai
import cv2

from host_bird_eye_view import BirdsEyeView
from host_rgb_conference_node import CombineOutputs
from host_display import Display


device = dai.Device()
device.setIrLaserDotProjectorIntensity(1)
platform = device.getPlatform()
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    OUTPUT_SHAPE = (512, 288)
    FPS = 10 if platform == dai.Platform.RVC2 else 30

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    cam.initialControl.setManualFocus(130)
    cam_output = cam.requestOutput(OUTPUT_SHAPE, type=dai.ImgFrame.Type.NV12, fps=FPS)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(OUTPUT_SHAPE, type=dai.ImgFrame.Type.NV12, fps=FPS),
        right=right.requestOutput(OUTPUT_SHAPE, type=dai.ImgFrame.Type.NV12, fps=FPS),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
    )
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)
    if platform == dai.Platform.RVC2:
        stereo.setOutputSize(*OUTPUT_SHAPE)

    spatialDetectionNetwork = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        cam,
        stereo,
        dai.NNModelDescription(
            modelSlug="yolov6-nano", modelVersionSlug="r2-coco-512x288"
        ),
        fps=FPS,
    )
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(300)
    spatialDetectionNetwork.setDepthUpperThreshold(35000)
    spatialDetectionNetwork.input.setMaxSize(1)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.inputDepth.setMaxSize(1)
    spatialDetectionNetwork.inputDepth.setBlocking(False)

    # In order to not loose synced messages (on low bandwidth), do all syncing on device
    sync = pipeline.create(dai.node.Sync)
    sync.setRunOnHost(False)
    sync_color_input = sync.inputs["color"]
    sync_color_input.setBlocking(True)
    cam_output.link(sync_color_input)
    sync_depth_input = sync.inputs["depth"]
    sync_depth_input.setBlocking(False)
    spatialDetectionNetwork.passthroughDepth.link(sync_depth_input)
    sync_detections_input = sync.inputs["detections"]
    sync_detections_input.setMaxSize(1)
    sync_detections_input.setBlocking(False)
    spatialDetectionNetwork.out.link(sync_detections_input)

    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)

    bird_eye = pipeline.create(BirdsEyeView).build(demux.outputs["detections"])

    combined = pipeline.create(CombineOutputs).build(
        color=demux.outputs["color"],
        depth=demux.outputs["depth"],
        birdseye=bird_eye.output,
        detections=demux.outputs["detections"],
    )
    display = pipeline.create(Display).build(combined.output)
    display.setName("Luxonis")

    cv2.namedWindow("Luxonis", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Luxonis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Pipeline created.")
    pipeline.run()
