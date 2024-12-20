import depthai as dai
from host_node.normalize_bbox import NormalizeBbox
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display


device = dai.Device()
platform = device.getPlatform()
if platform == dai.Platform.RVC4:
    raise RuntimeError("ImgResizeMode.LETTERBOX is not yet supported on RVC4 platform")

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=25)
    cam_letterbox = cam.requestOutput(size=(512, 288), type=dai.ImgFrame.Type.BGR888i if platform == dai.Platform.RVC4 else dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.LETTERBOX, fps=25)

    nn = pipeline.create(dai.node.DetectionNetwork).build(cam_letterbox, nn_archive, confidenceThreshold=0.5)
    nn_classes = nn.getClasses()

    norm_isp = pipeline.create(NormalizeBbox).build(frame=cam_isp, nn=nn.out, manip_mode=dai.ImgResizeMode.LETTERBOX)
    dets_isp = pipeline.create(DrawDetections).build(frame=cam_isp, nn=norm_isp.output, label_map=nn_classes)

    norm_letterbox = pipeline.create(NormalizeBbox).build(frame=cam_letterbox, nn=nn.out, manip_mode=dai.ImgResizeMode.LETTERBOX)
    dets_letterbox = pipeline.create(DrawDetections).build(frame=cam_letterbox, nn=norm_letterbox.output, label_map=nn_classes)
    
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")
    
    display_letterbox = pipeline.create(Display).build(frames=dets_letterbox.output)
    display_letterbox.setName("Letterboxed")
    
    print("Pipeline created.")
    pipeline.run()
