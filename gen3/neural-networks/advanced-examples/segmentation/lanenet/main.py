import depthai as dai
import argparse

from os.path import isfile
from pathlib import Path
from host_lanenet import LaneNet
from download import download_vids

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', default="vids/vid3.mp4", type=str
                    , help="Path to video to use for inference. Default: vids/vid3.mp4")
args = parser.parse_args()

nn_shape = (512, 256)

# Download test videos
if not isfile(Path("vids/vid1.mp4").resolve().absolute()) or \
    not isfile(Path("vids/vid2.mp4").resolve().absolute()) or \
    not isfile(Path("vids/vid3.mp4").resolve().absolute()):
    download_vids()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setSize(*nn_shape)
    replay.setReplayVideoFile(Path(args.video).resolve().absolute())
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(Path("model/lanenet_openvino_2021.4_6shave.blob").resolve().absolute())
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    replay.out.link(nn.input)

    lane_detection = pipeline.create(LaneNet).build(
        preview=replay.out,
        nn=nn.out,
        nn_shape=nn_shape
    )
    lane_detection.inputs["preview"].setBlocking(False)
    lane_detection.inputs["preview"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
