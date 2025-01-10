import depthai as dai
import argparse
from host_stereo_matching import StereoMatching
from pathlib import Path
from download import download_blobs
from os.path import isfile

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn",
    "--nn-choice",
    type=str,
    help="Choose between 2 neural network models from {120x160,160x240} (the bigger one is default)",
)
args = parser.parse_args()

if args.nn_choice is None or args.nn_choice == "160x240":
    nn_path = (
        Path("models/crestereo_init_iter2_160x240_6_shaves.blob").resolve().absolute()
    )
    nn_shape = (160, 240)
elif args.nn_choice == "120x160":
    nn_path = (
        Path("models/crestereo_init_iter2_120x160_6_shaves.blob").resolve().absolute()
    )
    nn_shape = (120, 160)
else:
    nn_path = None
    nn_shape = None

# Download neural network models
if not isfile(
    Path("models/crestereo_init_iter2_160x240_6_shaves.blob").resolve().absolute()
) or not isfile(
    Path("models/crestereo_init_iter2_120x160_6_shaves.blob").resolve().absolute()
):
    download_blobs()

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    left = pipeline.create(dai.node.MonoCamera)
    left.setFps(2)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setFps(2)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(nn_path)

    manip_left = pipeline.create(dai.node.ImageManip)
    manip_left.initialConfig.setResize(nn_shape[1], nn_shape[0])
    manip_left.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    stereo.rectifiedLeft.link(manip_left.inputImage)

    manip_right = pipeline.create(dai.node.ImageManip)
    manip_right.initialConfig.setResize(nn_shape[1], nn_shape[0])
    manip_right.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    stereo.rectifiedRight.link(manip_right.inputImage)

    manip_left.out.link(nn.inputs["left"])
    manip_right.out.link(nn.inputs["right"])

    classification = pipeline.create(StereoMatching).build(
        disparity=stereo.disparity,
        nn=nn.out,
        nn_shape=nn_shape,
        max_disparity=stereo.initialConfig.getMaxDisparity(),
    )
    classification.inputs["disparity"].setBlocking(True)
    classification.inputs["nn"].setBlocking(True)

    print("Pipeline created.")
    pipeline.run()
