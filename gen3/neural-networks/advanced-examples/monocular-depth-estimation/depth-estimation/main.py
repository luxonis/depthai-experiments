import argparse

import cv2
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from host_node.host_depth_color_transform import DepthColorTransform
from host_node.overlay_frames import OverlayFrames
from host_node.visualize_detections_v2 import VisualizeDetectionsV2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn",
    "--neural-network",
    type=str,
    choices=[
        "midas_small",
        "midas_medium",
        "midas_large",
        "midas_xlarge",
        "midas_xxlarge",
    ],
    default="midas_small",
    help="Choose the neural network model used for depth estimation (midas_small is default)",
)
parser.add_argument(
    "-fps",
    "--frames-per-second",
    type=int,
    default=10,
    help="Choose the number of frames per second (10 is default)",
)
args = parser.parse_args()

nn_configs = {
    "midas_small": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-192x256",
        "size": (256, 192),
    },
    "midas_medium": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-256x384",
        "size": (384, 256),
    },
    "midas_large": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-288x512",
        "size": (512, 288),
    },
    "midas_xlarge": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-384x512",
        "size": (512, 384),
    },
    "midas_xxlarge": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-768x1024",
        "size": (1024, 768),
    },
}

selected_nn_config = nn_configs[args.neural_network]

device = dai.Device()

model_description = dai.NNModelDescription(
    modelSlug=selected_nn_config["model_slug"],
    platform=device.getPlatform().name,
    modelVersionSlug=selected_nn_config["model_version_slug"],
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

VIDEO_SIZE = (1280, 720)
NN_SIZE = selected_nn_config["size"]
FPS = args.frames_per_second

visualizer = dai.RemoteConnection()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    color_out = cam.requestOutput(
        size=VIDEO_SIZE, type=dai.ImgFrame.Type.BGR888p, fps=FPS
    )

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*NN_SIZE)
    color_out.link(nn_resize.inputImage)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=nn_resize.out, nn_source=nn_archive
    )

    visualize_detections = pipeline.create(VisualizeDetectionsV2).build(nn.out)
    color_transform = pipeline.create(DepthColorTransform).build(
        visualize_detections.output_mask
    )
    color_transform.setColormap(cv2.COLORMAP_INFERNO)

    overlay_resize = pipeline.create(dai.node.ImageManipV2)
    overlay_resize.initialConfig.addResize(*VIDEO_SIZE)
    overlay_resize.setMaxOutputFrameSize(VIDEO_SIZE[0] * VIDEO_SIZE[1] * 3)
    color_transform.output.link(overlay_resize.inputImage)

    overlay_frames = pipeline.create(OverlayFrames).build(color_out, overlay_resize.out)

    visualizer.addTopic("Color", color_out)
    visualizer.addTopic("Depth", overlay_resize.out)
    visualizer.addTopic("Depth overlay", overlay_frames.output)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
