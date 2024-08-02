import argparse
from pathlib import Path
import blobconverter
import depthai as dai
from host_efficientnet_classification import EfficientnetClassification

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-vid', '--video', type=str
                    , help="Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)")
args = parser.parse_args()

# NOTE: video must be of size 224 x 224. We will resize this on the
# host, but you could also use ImageManip node to do it on device

debug = not args.no_debug
camera = not args.video

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blobconverter.from_zoo(name="efficientnet-b0", zoo_type="depthai", shaves=6, version="2021.4"))

    if camera:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(224, 224)
        cam.setInterleaved(False)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        cam.preview.link(nn.input)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize((224, 224))
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)

        replay.out.link(nn.input)

    classification = pipeline.create(EfficientnetClassification).build(
        preview=cam.preview if camera else replay.out,
        nn=nn.out,
        debug=debug
    )

    print("Pipeline created.")
    pipeline.run()
