import argparse
from os.path import isfile
from pathlib import Path

import depthai as dai
import lane_detection_config
from depthai_nodes import LaneDetectionParser
from download import download_vids
from host_node.visualize_detections import VisualizeDetections

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
nn_configs = {
    "culane": lane_detection_config.culane,
    "tusimple": lane_detection_config.tusimple,
}

NN_SIZE = (800, 288)
VIDEO_SIZE = (800, 288)

device = dai.Device()

nn_model_description = dai.NNModelDescription(
    modelSlug="ultra-fast-lane-detection",
    platform=device.getPlatform().name,
    modelVersionSlug=nn_version_slugs[args.neural_network],
)
nn_archive_path = dai.getModelFromZoo(nn_model_description, useCached=True)
nn_archive = dai.NNArchive(nn_archive_path)

nn_config = nn_configs[args.neural_network]

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
        color_out = cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=5)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setSize(*VIDEO_SIZE)
        replay.setFps(5)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        color_out = replay.out

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*NN_SIZE)
    color_out.link(nn_resize.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.setNumInferenceThreads(2)
    nn_resize.out.link(nn.input)

    parser = pipeline.create(LaneDetectionParser)
    parser.setRowAnchors(nn_config["row_anchors"])
    parser.setGridingNum(nn_config["griding_num"])
    parser.setClsNumPerLane(nn_config["cls_num_per_lane"])
    parser.setInputShape(nn_config["input_shape"])
    parser.set
    nn.out.link(parser.input)

    visualize_detections = pipeline.create(VisualizeDetections).build(parser.out)

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
