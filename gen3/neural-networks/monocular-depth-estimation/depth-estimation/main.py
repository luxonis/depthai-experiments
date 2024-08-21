import depthai as dai
import argparse

from pathlib import Path
from host_depth_estimation import DepthEstimation

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--neural-network', type=str
                    , choices=['fast_small', 'fast_large', 'mbnv_small', 'mbnv_large', 'mega'], default='mega'
                    , help="Choose the neural network model used for depth estimation (fast_small is default)")
args = parser.parse_args()

nn_path = None
nn_shape = None

if args.neural_network == "fast_small":
    model_description = dai.NNModelDescription(modelSlug="fast-depth", platform="RVC2", modelVersionSlug="320x256")
    nn_shape = (320, 256)
elif args.neural_network == "fast_large":
    model_description = dai.NNModelDescription(modelSlug="fast-depth", platform="RVC2", modelVersionSlug="640x480")
    nn_shape = (640, 480)
elif args.neural_network == "mbnv_small":
    model_description = dai.NNModelDescription(modelSlug="depth-estimation", platform="RVC2", modelVersionSlug="320x240")
    nn_shape = (320, 240)
elif args.neural_network == "mbnv_large":
    model_description = dai.NNModelDescription(modelSlug="depth-estimation", platform="RVC2", modelVersionSlug="640x480")
    nn_shape = (640, 480)
elif args.neural_network == "mega":
    raise NotImplementedError("Mega model is not supported yet")
    nn_path = Path("../depth-estimation/model/megadepth_192x256_openvino_2021.4_6shave.blob").absolute().resolve()
    nn_shape = (256, 192)

archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    rgb_preview = cam.requestOutput(size=nn_shape, type=dai.ImgFrame.Type.BGR888p)
    
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    rgb_preview.link(nn.input)

    depth = pipeline.create(DepthEstimation).build(
        preview=rgb_preview,
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
