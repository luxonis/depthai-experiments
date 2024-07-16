import depthai as dai
import argparse
from pathlib import Path
from host_multiclass_segmentation import MulticlassSegmentation

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", choices=["left", "rgb", "right"], default="rgb", type=str
                    , help="Choose camera for inference source (rgb is default)")
args = parser.parse_args()

cam_source = args.cam_input
nn_shape = (256, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn_path = Path("models/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob").resolve().absolute()
    nn.setBlobPath(nn_path)
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
        cam = pipeline.create(dai.node.ColorCamera).build()
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

    multiclass_segmentation = pipeline.create(MulticlassSegmentation).build(
        preview=nn.passthrough,
        nn=nn.out,
        nn_shape=nn_shape
    )

    print("Pipeline created.")
    pipeline.run()
