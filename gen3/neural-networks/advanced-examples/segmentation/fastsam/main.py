import time

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from host_node.host_depth_color_transform import DepthColorTransform
from host_node.overlay_frames import OverlayFrames
from host_node.visualize_detections_v2 import VisualizeDetectionsV2

device = dai.Device()

segmentation_model_description = dai.NNModelDescription(
    modelSlug="fastsam-s",
    platform=device.getPlatform().name,
    modelVersionSlug="512x288",
)
segmentation_archive_path = dai.getModelFromZoo(segmentation_model_description)
segmentation_archive = dai.NNArchive(segmentation_archive_path)


VIDEO_SIZE = (1280, 720)
NN_SIZE = (512, 288)

FPS = 5 if device.getPlatform() == dai.Platform.RVC2 else 30

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = cam.requestOutput(
        size=VIDEO_SIZE, type=dai.ImgFrame.Type.BGR888p, fps=FPS
    )

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*NN_SIZE)
    color_out.link(nn_resize.inputImage)

    segmentation_nn = pipeline.create(ParsingNeuralNetwork).build(
        input=nn_resize.out, nn_source=segmentation_archive
    )

    visualize_detections = pipeline.create(VisualizeDetectionsV2).build(
        segmentation_nn.out
    )
    visualize_detections.set_color((0, 255, 0))

    color_transform = pipeline.create(DepthColorTransform).build(
        visualize_detections.output_mask
    )

    mask_resize = pipeline.create(dai.node.ImageManipV2)
    mask_resize.initialConfig.addResize(*VIDEO_SIZE)
    mask_resize.setMaxOutputFrameSize(VIDEO_SIZE[0] * VIDEO_SIZE[1] * 3)
    color_transform.output.link(mask_resize.inputImage)

    overlay_frames = pipeline.create(OverlayFrames).build(color_out, mask_resize.out)
    overlay_frames.set_weigths(0.7, 0.3)

    visualizer.addTopic("Camera", color_out)
    visualizer.addTopic("Mask", overlay_frames.output)
    visualizer.addTopic("Outlines", visualize_detections.output)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        time.sleep(0.01)
    print("Pipeline finished.")
