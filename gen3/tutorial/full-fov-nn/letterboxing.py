import depthai as dai
from host_nodes.host_add_detections import NormaliezeBbox, DrawDetections
from host_nodes.host_display import Display

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)


with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.BGR888p, fps=25)
    cam_letterbox = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.LETTERBOX)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    nn_classes = nn.getClasses()

    cam_letterbox.link(nn.input)

    norm_isp = pipeline.create(NormaliezeBbox).build(frame=cam_isp, nn=nn.out, manip_mode=dai.ImgResizeMode.LETTERBOX)
    dets_isp = pipeline.create(DrawDetections).build(frame=cam_isp, nn=norm_isp.output, label_map=nn_classes)

    norm_letterbox = pipeline.create(NormaliezeBbox).build(frame=cam_letterbox, nn=nn.out, manip_mode=dai.ImgResizeMode.LETTERBOX)
    dets_letterbox = pipeline.create(DrawDetections).build(frame=cam_letterbox, nn=norm_letterbox.output, label_map=nn_classes)
    
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")
    
    display_letterbox = pipeline.create(Display).build(frames=dets_letterbox.output)
    display_letterbox.setName("Letterboxed")
    
    print("Pipeline created.")
    pipeline.run()
