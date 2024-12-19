import depthai as dai

from host_node.normalize_bbox import NormalizeBbox
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display


device = dai.Device()
platform = device.getPlatform()

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath=archive_path)


with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=25, resizeMode=dai.ImgResizeMode.STRETCH)
    cam_stretch = cam.requestOutput(size=(512, 288), type=dai.ImgFrame.Type.BGR888i if platform == dai.Platform.RVC4 else dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.STRETCH, fps=25)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    nn_classes = nn.getClasses()

    cam_stretch.link(nn.input)

    norm_isp = pipeline.create(NormalizeBbox).build(frame=cam_isp, nn=nn.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_isp = pipeline.create(DrawDetections).build(frame=cam_isp, nn=norm_isp.output, label_map=nn_classes)

    norm_stretch = pipeline.create(NormalizeBbox).build(frame=cam_stretch, nn=nn.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_stretch = pipeline.create(DrawDetections).build(frame=cam_stretch, nn=norm_stretch.output, label_map=nn_classes)
    
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")
    
    display_crop = pipeline.create(Display).build(frames=dets_stretch.output)
    display_crop.setName("Stretched")
    
    print("Pipeline created.")
    pipeline.run()
