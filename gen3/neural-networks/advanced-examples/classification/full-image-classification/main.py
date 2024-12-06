import argparse
from pathlib import Path

import depthai as dai
from depthai_nodes import ClassificationParser
from efficientnet_classes import CLASS_NAMES
from host_node.visualize_detections import VisualizeDetections

parser = argparse.ArgumentParser()
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)",
)
args = parser.parse_args()


device = dai.Device()

nn_description = dai.NNModelDescription(
    modelSlug="efficientnet-lite",
    platform=device.getPlatform().name,
    modelVersionSlug="lite0-224x224",
)
nn_path = dai.getModelFromZoo(nn_description, useCached=True)
nn_archive = dai.NNArchive(archivePath=nn_path)

NN_SIZE = (224, 224)

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(1280, 720)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        color_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.BGR888p)

    manip = pipeline.create(dai.node.ImageManipV2)
    manip.initialConfig.addResize(*NN_SIZE)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    color_out.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    manip.out.link(nn.input)

    parser = pipeline.create(ClassificationParser)
    parser.setClasses(CLASS_NAMES)
    nn.out.link(parser.input)

    visualize_detections = pipeline.create(VisualizeDetections).build(nn=parser.out)

    visualizer.addTopic("Classification", visualize_detections.output)
    visualizer.addTopic("Color", color_out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
