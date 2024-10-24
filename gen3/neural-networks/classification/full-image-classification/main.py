import argparse
from pathlib import Path

import depthai as dai
from host_image_classification import ImageClassification

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn",
    "--neural-network",
    choices=["efficientnet", "flowers"],
    default="efficientnet",
    type=str,
    help="Choose the neural network model used for classification (efficientnet is default)",
)
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)",
)
args = parser.parse_args()

nn_shape = None
blob = None

device = dai.Device()

if args.neural_network == "efficientnet":
    device_platform = device.getPlatform()
    if device_platform == dai.Platform.RVC2:
        model_description = dai.NNModelDescription(
            modelSlug="efficientnet-lite",
            platform=device_platform.name,
            modelVersionSlug="lite0-224x224",
        )
        nn_shape = (224, 224)
    else:
        model_description = dai.NNModelDescription(
            modelSlug="efficientnet",
            platform=device_platform.name,
            modelVersionSlug="300x300",
        )
        nn_shape = (300, 300)
    archive_path = dai.getModelFromZoo(model_description)
    blob = dai.NNArchive(archive_path).getSuperBlob().getBlobWithNumShaves(6)
elif args.neural_network == "flowers":
    nn_shape = (180, 180)
    blob = dai.OpenVINO.Blob(Path("model/flower.blob").resolve().absolute())

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(400, 400)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        preview = replay.out
    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(400, 400)
        cam.setInterleaved(False)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        preview = cam.preview

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*nn_shape)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlob(blob)
    manip.out.link(nn.input)

    classification = pipeline.create(ImageClassification).build(
        preview=preview, nn=nn.out, efficientnet=(args.neural_network == "efficientnet")
    )
    classification.inputs["preview"].setBlocking(False)
    classification.inputs["preview"].setMaxSize(4)
    classification.inputs["nn"].setBlocking(False)
    classification.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
