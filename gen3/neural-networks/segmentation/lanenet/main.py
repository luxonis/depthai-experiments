import argparse
from os.path import isfile
from pathlib import Path

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from download import download_vids
from host_node.visualize_detections_v2 import VisualizeDetectionsV2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--video",
    default="vids/vid3.mp4",
    type=str,
    help="Path to video to use for inference. Default: vids/vid3.mp4",
)
parser.add_argument(
    "-cam", "--cam", help="Use camera instead of video.", action="store_true"
)
parser.add_argument(
    "-nn",
    "--neural-network",
    choices=["culane", "tusimple"],
    default="tusimple",
    type=str,
    help="Choose the neural network model (tusimple is default)",
)

args = parser.parse_args()

nn_version_slugs = {
    "culane": "culane-800x288",
    "tusimple": "tusimple-800x288",
}

NN_SIZE = (800, 288)
VIDEO_SIZE = (800, 288)
FPS = 10

device = dai.Device()

nn_model_description = dai.NNModelDescription(
    modelSlug="ultra-fast-lane-detection",
    platform=device.getPlatform().name,
    modelVersionSlug=nn_version_slugs[args.neural_network],
)
nn_archive_path = dai.getModelFromZoo(nn_model_description, useCached=True)
nn_archive = dai.NNArchive(nn_archive_path)


# Download test videos
if (
    not isfile(Path("vids/vid1.mp4").resolve().absolute())
    or not isfile(Path("vids/vid2.mp4").resolve().absolute())
    or not isfile(Path("vids/vid3.mp4").resolve().absolute())
):
    download_vids()

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.cam:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=FPS)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setSize(*VIDEO_SIZE)
        replay.setFps(FPS)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        color_out = replay.out

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*NN_SIZE)
    color_out.link(nn_resize.inputImage)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=nn_resize.out, nn_source=nn_archive
    )

    visualize_detections = pipeline.create(VisualizeDetectionsV2).build(nn=nn.out)

    print("Pipeline created.")
    visualizer.addTopic("Lines", visualize_detections.output)
    visualizer.addTopic("Color", nn_resize.out)
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
