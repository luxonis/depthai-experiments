import argparse
from os.path import isfile
from pathlib import Path

import depthai as dai
from download import download_vids
from host_lanenet import LaneNet

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
args = parser.parse_args()

NN_SIZE = (512, 256)
VIDEO_SIZE = (1280, 720)

# Download test videos
if (
    not isfile(Path("vids/vid1.mp4").resolve().absolute())
    or not isfile(Path("vids/vid2.mp4").resolve().absolute())
    or not isfile(Path("vids/vid3.mp4").resolve().absolute())
):
    download_vids()

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    if args.cam:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=10)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setSize(*NN_SIZE)
        replay.setFps(10)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        color_out = replay.out

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*NN_SIZE)
    color_out.link(nn_resize.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(
        Path("model/lanenet_openvino_2021.4_6shave.blob").resolve().absolute()
    )
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    nn_resize.out.link(nn.input)

    lane_detection = pipeline.create(LaneNet).build(
        preview=nn_resize.out, nn=nn.out, nn_shape=NN_SIZE
    )
    lane_detection.inputs["preview"].setBlocking(False)
    lane_detection.inputs["preview"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
