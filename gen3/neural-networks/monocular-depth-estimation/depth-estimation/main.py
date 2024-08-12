import depthai as dai
import argparse
import blobconverter

from pathlib import Path
from host_depth_estimation import DepthEstimation

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--neural-network', type=str
                    , choices=['fast_small', 'fast_large', 'mbnv_small', 'mbnv_large', 'mega'], default='fast_small'
                    , help="Choose the neural network model used for depth estimation (fast_small is default)")
args = parser.parse_args()

nn_path = None
nn_shape = None

if args.neural_network == "fast_small":
    nn_path = blobconverter.from_zoo(name="fast_depth_256x320", zoo_type="depthai")
    nn_shape = (256, 320)
elif args.neural_network == "fast_large":
    nn_path = blobconverter.from_zoo(name="fast_depth_480x640", zoo_type="depthai")
    nn_shape = (480, 640)
elif args.neural_network == "mbnv_small":
    nn_path = blobconverter.from_zoo(name="depth_estimation_mbnv2_240x320", zoo_type="depthai")
    nn_shape = (240, 320)
elif args.neural_network == "mbnv_large":
    nn_path = blobconverter.from_zoo(name="depth_estimation_mbnv2_480x640", zoo_type="depthai")
    nn_shape = (480, 640)
elif args.neural_network == "mega":
    nn_path = Path("../depth-estimation/model/megadepth_192x256_openvino_2021.4_6shave.blob").absolute().resolve()
    nn_shape = (192, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(nn_shape[1], nn_shape[0])
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(nn_path)
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    cam.preview.link(nn.input)

    depth = pipeline.create(DepthEstimation).build(
        preview=cam.preview,
        nn=nn.out,
        nn_shape=nn_shape,
        mbnv2=(args.neural_network == "mbnv_small" or args.neural_network == "mbnv_large")
    )
    depth.inputs["preview"].setBlocking(False)
    depth.inputs["preview"].setMaxSize(4)
    depth.inputs["nn"].setBlocking(False)
    depth.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
