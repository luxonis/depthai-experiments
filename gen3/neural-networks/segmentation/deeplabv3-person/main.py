import depthai as dai
import argparse
import blobconverter
from host_person_segmentation import PersonSegmentation

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", choices=["256", "513"], default="256", type=str
                    , help="Choose between 2 sizes of the neural network (the smaller one is default)")
parser.add_argument("-cam", "--cam_input", choices=["left", "rgb", "right"], default="rgb", type=str
                    , help="Choose camera for inference source (rgb is default)")
args = parser.parse_args()

cam_source = args.cam_input
nn_shape = (int(args.size), int(args.size))

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn_path = blobconverter.from_zoo(name=f"deeplab_v3_mnv2_{args.size}x{args.size}", zoo_type="depthai", shaves=6)
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

    person_segmentation = pipeline.create(PersonSegmentation).build(
        preview=nn.passthrough,
        nn=nn.out,
        nn_shape=nn_shape
    )

    print("Pipeline created.")
    pipeline.run()
