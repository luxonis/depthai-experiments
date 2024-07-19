import depthai as dai
import argparse
import blobconverter
from host_depth_mbnv2 import DepthMBNV2

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn-choice', type=str
                    , help="Choose between 2 neural network models from {240x320,480x640} (the smaller one is default)")
args = parser.parse_args()

if args.nn_choice is None or args.nn_choice == "240x320":
    nn_path = blobconverter.from_zoo(name="depth_estimation_mbnv2_240x320", zoo_type="depthai", shaves=6)
    nn_shape = (240, 320)
elif args.nn_choice == "480x640":
    nn_path = blobconverter.from_zoo(name="depth_estimation_mbnv2_480x640", zoo_type="depthai", shaves=6)
    nn_shape = (480, 640)
else:
    nn_path = None
    nn_shape = None
    fps = None

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(nn_shape[1], nn_shape[0])
    cam.setInterleaved(False)
    cam.setFps(20)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(nn_path)
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    cam.preview.link(nn.input)

    depth = pipeline.create(DepthMBNV2).build(
        preview=cam.preview,
        nn=nn.out,
        nn_shape=nn_shape
    )
    depth.inputs["preview"].setBlocking(False)
    depth.inputs["preview"].setMaxSize(4)
    depth.inputs["nn"].setBlocking(False)
    depth.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
