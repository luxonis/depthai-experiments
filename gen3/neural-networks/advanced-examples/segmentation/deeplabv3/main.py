import argparse

import cv2
import depthai as dai
from depthai_nodes import SegmentationParser
from host_node.host_depth_color_transform import DepthColorTransform
from host_node.overlay_frames import OverlayFrames

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn",
    "--neural-network",
    choices=["multiclass", "person_small", "person_large"],
    default="person_small",
    type=str,
    help="Choose the neural network model used for segmentation (multiclass is default)",
)

version_slugs = {
    "multiclass": "513x513",
    "person_small": "person-256x256",
    "person_large": "person-513x513",
}
nn_sizes = {
    "multiclass": (513, 513),
    "person_small": (256, 256),
    "person_large": (513, 513),
}
num_classes = {
    "multiclass": 21,
    "person_small": 2,
    "person_large": 2,
}


VIDEO_SIZE = (720, 720)

args = parser.parse_args()
version_slug = version_slugs[args.neural_network]
nn_size = nn_sizes[args.neural_network]
classes = num_classes[args.neural_network]

device = dai.Device()

nn_model_description = dai.NNModelDescription(
    modelSlug="deeplab-v3-plus",
    platform=device.getPlatform().name,
    modelVersionSlug=version_slug,
)
nn_archive_path = dai.getModelFromZoo(nn_model_description)
nn_archive = dai.NNArchive(nn_archive_path)

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=10)

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*nn_size)
    color_out.link(nn_resize.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn_resize.out.link(nn.input)

    parser = pipeline.create(SegmentationParser)
    parser.setBackgroundClass(True)
    nn.out.link(parser.input)

    color_transform = pipeline.create(DepthColorTransform).build(parser.out)
    color_transform.setMaxDisparity(classes)
    color_transform.setColormap(cv2.COLORMAP_JET)

    segmentation_map_resize = pipeline.create(dai.node.ImageManipV2)
    segmentation_map_resize.initialConfig.addResize(*VIDEO_SIZE)
    segmentation_map_resize.setMaxOutputFrameSize(VIDEO_SIZE[0] * VIDEO_SIZE[1] * 3)
    color_transform.output.link(segmentation_map_resize.inputImage)

    overlay_frames = pipeline.create(OverlayFrames).build(
        color_out, segmentation_map_resize.out
    )

    print("Pipeline created.")
    visualizer.addTopic("Segmentation", overlay_frames.output)
    visualizer.addTopic("Color", color_out)
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
