import depthai as dai
import argparse

from host_human_pose import HumanPose
from host_fps_drawer import FPSDrawer
from display import Display
from pathlib import Path

model_description = dai.NNModelDescription(modelSlug="human-pose-estimation", platform="RVC2", modelVersionSlug="0001-456x256")
archive_path = dai.getModelFromZoo(model_description)

parser = argparse.ArgumentParser()
parser.add_argument('-vid', '--video', type=str
                    , help="Path to video file to be used for inference (otherwise uses the DepthAI 4K color camera)")
args = parser.parse_args()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(456, 256)
    manip.initialConfig.setKeepAspectRatio(False)

    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.out.link(manip.inputImage)

    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.requestOutput((768, 432), dai.ImgFrame.Type.BGR888p).link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork).build(
        output=manip.out,
        nnArchive=dai.NNArchive(archive_path)
    )
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    human_pose = pipeline.create(HumanPose).build(
        preview=replay.out if args.video else cam.requestOutput((768, 432), dai.ImgFrame.Type.BGR888p),
        nn=nn.out
    )
    human_pose.inputs["preview"].setBlocking(False)
    human_pose.inputs["preview"].setMaxSize(2)
    human_pose.inputs["nn"].setBlocking(False)
    human_pose.inputs["nn"].setMaxSize(2)

    fps_drawer = pipeline.create(FPSDrawer).build(human_pose.output)

    display = pipeline.create(Display).build(fps_drawer.output)
    display.set_window_name("Preview")

    print("Pipeline created.")
    pipeline.run()
