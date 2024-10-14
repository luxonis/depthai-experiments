import depthai as dai
from depthai_nodes.ml.parsers import YuNetParser
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.normalize_detections import NormalizeDetections
from host_node.parser_bridge import ParserBridge

device = dai.Device()

model_description = dai.NNModelDescription(
    modelSlug="yunet",
    platform=device.getPlatform().name,
    modelVersionSlug="640x640",
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    preview = cam.requestOutput((640, 640), type=dai.ImgFrame.Type.BGR888p, fps=20)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setNNArchive(nn_archive)
    preview.link(detection_nn.input)
    yunet_parser = pipeline.create(YuNetParser)
    detection_nn.out.link(yunet_parser.input)
    bridge = pipeline.create(ParserBridge).build(nn=yunet_parser.out)

    norm_det = pipeline.create(NormalizeDetections).build(
        frame=preview, nn=yunet_parser.out
    )
    draw_detections = pipeline.create(DrawDetections).build(
        frame=preview, nn=norm_det.output, label_map=["face"]
    )
    draw_detections.set_draw_labels(False)

    display = pipeline.create(Display).build(frames=draw_detections.output)
    display.setName("Face detection")

    print("Pipeline created.")
    pipeline.run()
