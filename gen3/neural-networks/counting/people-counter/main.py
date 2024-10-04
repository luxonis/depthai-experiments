import argparse
from pathlib import Path

import depthai as dai
from depthai_nodes.ml.parsers import SCRFDParser
from host_node.counter_text_serializer import CounterTextSerializer
from host_node.draw_detections import DrawDetections
from host_node.draw_text import DrawText
from host_node.host_display import Display
from host_node.normalize_bbox import NormalizeBbox
from host_node.object_counter import ObjectCounter
from host_node.parser_bridge import ParserBridge

parser = argparse.ArgumentParser()
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video to use for inference. Otherwise uses the DepthAI color camera",
)
args = parser.parse_args()

device = dai.Device()
model_description = dai.NNModelDescription(
    modelSlug="scrfd-person-detection",
    platform=device.getPlatform().name,
    modelVersionSlug="2-5g-640x640",
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(640, 640)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        frame_type = pipeline.create(dai.node.ImageManip)
        frame_type.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        frame_type.setMaxOutputFrameSize(640 * 640 * 3)
        replay.out.link(frame_type.inputImage)

        preview = frame_type.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        preview = cam.requestOutput(size=(640, 640), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.input.setBlocking(False)
    preview.link(nn.input)
    nn_parser = pipeline.create(SCRFDParser)
    nn_parser.setFeatStrideFPN((8, 16, 32, 64, 128))
    nn_parser.setNumAnchors(1)
    nn.out.link(nn_parser.input)
    bridge = pipeline.create(ParserBridge).build(nn=nn_parser.out)

    object_counter = pipeline.create(ObjectCounter).build(nn=bridge.output)
    counter_serializer = pipeline.create(CounterTextSerializer).build(
        counter=object_counter.output, label_map=["People count"]
    )
    draw_count = pipeline.create(DrawText).build(
        frame=preview, text=counter_serializer.output
    )
    draw_count.config.position = (15, 30)
    draw_count.config.size = 1

    normalize_bbox = pipeline.create(NormalizeBbox).build(
        frame=draw_count.output, nn=bridge.output
    )
    draw_detections = pipeline.create(DrawDetections).build(
        frame=draw_count.output, nn=normalize_bbox.output, label_map=["Person"]
    )
    display = pipeline.create(Display).build(frames=draw_detections.output)

    print("Pipeline created.")
    pipeline.run()
