import argparse
from pathlib import Path
import depthai as dai
from host_image_classification import ImageClassification

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-vid', '--video', type=str
                    , help="Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)")
args = parser.parse_args()

debug = not args.no_debug
camera = not args.video

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(Path("flower.blob").resolve().absolute())

    if camera:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(480, 480)
        cam.setInterleaved(False)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(180, 180)

        cam.preview.link(manip.inputImage)
        manip.out.link(nn.input)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize((180, 180))
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)

        replay.out.link(nn.input)

    classification = pipeline.create(ImageClassification).build(
        preview=cam.preview if camera else replay.out,
        nn=nn.out,
        debug=debug
    )

    print("Pipeline created.")
    pipeline.run()
