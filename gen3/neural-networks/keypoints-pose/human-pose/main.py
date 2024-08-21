import depthai as dai
import blobconverter
import argparse

from host_human_pose import HumanPose
from display import Display
from pathlib import Path

model_description = dai.NNModelDescription(modelSlug="human-pose-estimation", platform="RVC2", modelVersionSlug="0001-456x256")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

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
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setIspScale(1, 5)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.initialControl.setManualFocus(130)
        cam.setInterleaved(False)
        cam.setPreviewSize(768, 432)

        cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    manip.out.link(nn.input)

    human_pose = pipeline.create(HumanPose).build(
        preview=replay.out if args.video else cam.preview,
        nn=nn.out
    )
    human_pose.inputs["preview"].setBlocking(False)
    human_pose.inputs["preview"].setMaxSize(2)
    human_pose.inputs["nn"].setBlocking(False)
    human_pose.inputs["nn"].setMaxSize(2)

    display = pipeline.create(Display).build(
        human_pose.output
    )
    display.set_window_name("Preview")

    print("Pipeline created.")
    pipeline.run()
