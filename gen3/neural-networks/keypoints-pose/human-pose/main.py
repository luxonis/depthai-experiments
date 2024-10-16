import depthai as dai
import argparse

from host_human_pose import HumanPose
from host_fps_drawer import FPSDrawer
from host_display import Display
from pathlib import Path
from parsing.hrnet_parser import HRNetParser

device = dai.Device()
model_description = dai.NNModelDescription(modelSlug="lite-hrnet", platform=device.getPlatform().name, modelVersionSlug="18-coco-256x192")
archive_path = dai.getModelFromZoo(model_description)

parser = argparse.ArgumentParser()
parser.add_argument('-vid', '--video', type=str
                    , help="Path to video file to be used for inference (otherwise uses the DepthAI color camera)")
args = parser.parse_args()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(192, 256)
    manip.initialConfig.setKeepAspectRatio(False)

    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setSize(192*5, 256*5)
        video_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        video_out = cam.requestOutput((192*5, 256*5), dai.ImgFrame.Type.BGR888p)
    video_out.link(manip.inputImage)
    
    nn = pipeline.create(dai.node.NeuralNetwork).build(
        input=manip.out,
        nnArchive=dai.NNArchive(archive_path)
    )
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    parser = pipeline.create(HRNetParser)
    parser.setScoreThreshold(0.0) # Do not prune any keypoints
    nn.out.link(parser.input)

    human_pose = pipeline.create(HumanPose).build(
        preview=video_out,
        keypoints=parser.out
    )
    human_pose.inputs["preview"].setBlocking(False)
    human_pose.inputs["preview"].setMaxSize(2)
    human_pose.inputs["keypoints"].setBlocking(False)
    human_pose.inputs["keypoints"].setMaxSize(2)

    fps_drawer = pipeline.create(FPSDrawer).build(human_pose.output)

    display = pipeline.create(Display).build(fps_drawer.output)
    display.setName("Preview")

    print("Pipeline created.")
    pipeline.run()
