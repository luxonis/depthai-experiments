import depthai as dai

from host_node.host_display import Display
from host_node.normalize_bbox import NormalizeBbox
from host_node.draw_detections import DrawDetections


device = dai.Device()
platform = device.getPlatform()

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)


with dai.Pipeline(device) as pipeline:

    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam.requestOutput(size=(720, 720), type=dai.ImgFrame.Type.NV12)

    crop = pipeline.create(dai.node.ImageManipV2)
    crop.initialConfig.addResize(512, 288)
    crop.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i if platform == dai.Platform.RVC4 else dai.ImgFrame.Type.BGR888p)
    preview.link(crop.inputImage)

    nn = pipeline.create(dai.node.DetectionNetwork).build(crop.out, nn_archive, confidenceThreshold=0.5)
    label_map = nn.getClasses()
    
    rgb_dets_normalized = pipeline.create(NormalizeBbox).build(frame=preview, nn=nn.out)
    rgb_dets = pipeline.create(DrawDetections).build(frame=preview, nn=rgb_dets_normalized.output, label_map=label_map)
    rgb_dets.set_color((0, 0, 255))
    rgb_dets.set_thickness(4)
    rgb = pipeline.create(Display).build(frames=rgb_dets.output)
    rgb.setName("RGB")

    crop_dets_normalized = pipeline.create(NormalizeBbox).build(frame=crop.out, nn=nn.out)
    crop_dets = pipeline.create(DrawDetections).build(frame=crop.out, nn=crop_dets_normalized.output, label_map=label_map)
    crop = pipeline.create(Display).build(frames=crop_dets.output)
    crop.setName("Crop")
    
    print("Pipeline created.")
    pipeline.run()
