import depthai as dai
from host_node.normalize_bbox import NormalizeBbox
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display


device = dai.Device()
platform = device.getPlatform()

model_description = dai.NNModelDescription(
    modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288"
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(1352, 1024), type=dai.ImgFrame.Type.NV12, fps=25)
    cam_crop = cam.requestOutput(
        size=(512, 288),
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
        fps=25,
    )

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        cam_crop, nn_archive, confidenceThreshold=0.5
    )
    nn_classes = nn.getClasses()

    norm_isp = pipeline.create(NormalizeBbox).build(frame=cam_isp, nn=nn.out)
    dets_isp = pipeline.create(DrawDetections).build(
        frame=cam_isp, nn=norm_isp.output, label_map=nn_classes
    )
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")

    norm_crop = pipeline.create(NormalizeBbox).build(frame=cam_crop, nn=nn.out)
    dets_crop = pipeline.create(DrawDetections).build(
        frame=cam_crop, nn=norm_crop.output, label_map=nn_classes
    )
    display_crop = pipeline.create(Display).build(frames=dets_crop.output)
    display_crop.setName("Cropped")

    print("Pipeline created.")
    pipeline.run()
