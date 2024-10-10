import depthai as dai
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.normalize_bbox import NormalizeBbox

device = dai.Device()

modelDescription = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="r2-coco-512x288",
)
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)


with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    color_out = cam.requestOutput(
        size=(512, 288), type=dai.ImgFrame.Type.BGR888p, fps=40
    )

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        input=color_out, nnArchive=nn_archive
    )

    norm_bbox = pipeline.create(NormalizeBbox).build(frame=color_out, nn=nn.out)
    draw_detections = pipeline.create(DrawDetections).build(
        frame=color_out, nn=norm_bbox.output, label_map=nn.getClasses()
    )
    display = pipeline.create(Display).build(frames=draw_detections.output)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")
