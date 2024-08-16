import depthai as dai
import argparse

from pathlib import Path
from host_deeplab_segmentation import DeeplabSegmentation

parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--neural-network", choices=["multiclass", "person_small", "person_large"]
                    , default="person_small", type=str
                    , help="Choose the neural network model used for segmentation (multiclass is default)")
parser.add_argument("-cam", "--cam_input", choices=["left", "rgb", "right"], default="rgb", type=str
                    , help="Choose camera for inference source (rgb is default)")
args = parser.parse_args()

cam_source = args.cam_input
nn_shape = None
nn_path = None

if args.neural_network == "multiclass":
    raise NotImplementedError("Multiclass model is not supported yet")
    nn_shape = (256, 256)
    nn_path = Path("model/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob").resolve().absolute()
elif args.neural_network == "person_small":
    nn_shape = (256, 256)
    nn_path = Path("./model/deeplab_v3_mnv2_256x256.rvc2.tar.xz").resolve().absolute()
elif args.neural_network == "person_large":
    nn_shape = (513, 513)
    nn_path = Path("./model/deeplab_v3_mnv2_513x513.rvc2.tar.xz").resolve().absolute()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    nn = pipeline.create(dai.node.NeuralNetwork)
    # nn.setBlobPath(nn_path)
    nn.setNNArchive(dai.NNArchive(str(nn_path)))
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)

    if cam_source == "left":
        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    elif cam_source == "right":
        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(nn_shape)
        cam.setInterleaved(False)
        cam.preview.link(nn.input)

    cam.setFps(40)
    if cam_source != "rgb":
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(*nn_shape)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip.setKeepAspectRatio(True)
        cam.out.link(manip.inputImage)
        manip.out.link(nn.input)

    multiclass_segmentation = pipeline.create(DeeplabSegmentation).build(
        preview=nn.passthrough,
        nn=nn.out,
        nn_shape=nn_shape,
        multiclass=(args.neural_network == "multiclass")
    )
    multiclass_segmentation.inputs["preview"].setBlocking(False)
    multiclass_segmentation.inputs["preview"].setMaxSize(4)
    multiclass_segmentation.inputs["nn"].setBlocking(False)
    multiclass_segmentation.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
