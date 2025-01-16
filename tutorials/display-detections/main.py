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
nn_archive = dai.NNArchive(archivePath=archive_path)

with dai.Pipeline(device) as pipeline:
    cam_rgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    cam_nn = cam_rgb.requestOutput(
        size=(512, 288),
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
    )  # 16:9 aspect ratio
    same_aspect_ratio = cam_rgb.requestOutput(
        size=(640, 360), type=dai.ImgFrame.Type.NV12
    )  # 16:9 aspect ratio
    cam_full_fov = cam_rgb.requestOutput(
        size=(640, 480), type=dai.ImgFrame.Type.NV12
    )  # 4:3 aspect ratio

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        input=cam_nn, nnArchive=nn_archive
    )
    label_map = nn.getClasses()

    norm_full = pipeline.create(NormalizeBbox).build(frame=cam_full_fov, nn=nn.out)
    full_dets = pipeline.create(DrawDetections).build(
        frame=cam_full_fov, nn=norm_full.output, label_map=label_map
    )
    full = pipeline.create(Display).build(frames=full_dets.output)
    full.setName("Full cam FOV (4:3)")

    norm_square = pipeline.create(NormalizeBbox).build(
        frame=same_aspect_ratio, nn=nn.out
    )
    square_dets = pipeline.create(DrawDetections).build(
        frame=same_aspect_ratio, nn=norm_square.output, label_map=label_map
    )
    same_aspect_ratio = pipeline.create(Display).build(frames=square_dets.output)
    same_aspect_ratio.setName("Same aspect ratio (16:9)")

    norm_passthrough = pipeline.create(NormalizeBbox).build(frame=cam_nn, nn=nn.out)
    passthrough_dets = pipeline.create(DrawDetections).build(
        frame=cam_nn, nn=norm_passthrough.output, label_map=label_map
    )
    passthrough = pipeline.create(Display).build(frames=passthrough_dets.output)
    passthrough.setName("NN input (16:9)")

    print("Pipeline created.")
    pipeline.run()
