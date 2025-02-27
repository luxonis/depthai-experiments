import depthai as dai

from host_node.host_display import Display
from host_node.normalize_bbox import NormalizeBbox
from host_node.draw_detections import DrawDetections

device = dai.Device()
platform = device.getPlatform()

model_description = dai.NNModelDescription(
    modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288"
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    preview = cam.requestOutput(
        size=(512, 288),
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
    )

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        preview, nn_archive, confidenceThreshold=0.5
    )
    label_map = nn.getClasses()

    normalized_detections = pipeline.create(NormalizeBbox).build(
        frame=preview, nn=nn.out
    )
    draw_detections = pipeline.create(DrawDetections).build(
        frame=preview, nn=normalized_detections.output, label_map=label_map
    )
    rgb = pipeline.create(Display).build(frames=draw_detections.output)
    rgb.setName("RGB")

    print("Pipeline created.")
    pipeline.run()
