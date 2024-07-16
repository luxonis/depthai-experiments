from pathlib import Path
import depthai as dai
import argparse
from host_crowdcounting import Crowdcounting
from download import download_vids
from os.path import isfile

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video-path', type=str, help="Path to the video input for inference. Default: vids/virat.mp4"
                    , default="vids/virat.mp4")
args = parser.parse_args()

video_source = Path(args.video_path).resolve().absolute()
nn_shape = (426, 240)

# Download test videos
if not isfile(Path("vids/virat.mp4").resolve().absolute()) or \
    not isfile(Path("vids/vid1.mp4").resolve().absolute()) or \
    not isfile(Path("vids/vid2.mp4").resolve().absolute()) or \
    not isfile(Path("vids/vid3.mp4").resolve().absolute()) or \
    not isfile(Path("vids/vid4.mp4").resolve().absolute()):
    download_vids()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setReplayVideoFile(video_source)
    replay.setSize(nn_shape)
    replay.setLoop(False)
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
    replay.setFps(1)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResizeThumbnail(nn_shape)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.inputImage.setBlocking(True)
    replay.out.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(Path("model/vgg_openvino_2021.4_6shave.blob").resolve().absolute())
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    manip.out.link(nn.input)

    crowdcounting = pipeline.create(Crowdcounting).build(
        preview=manip.out,
        nn=nn.out,
        nn_shape=nn_shape
    )

    print("Pipeline created.")
    pipeline.run()
