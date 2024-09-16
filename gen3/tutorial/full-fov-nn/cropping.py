import depthai as dai
from host_nodes.host_add_detections import NormaliezeBbox, DrawDetections
from host_nodes.host_display import Display

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)


with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(1352, 1024), type=dai.ImgFrame.Type.BGR888p, fps=25)
    cam_crop = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    nn_classes = nn.getClasses()

    cam_crop.link(nn.input)

    norm_isp = pipeline.create(NormaliezeBbox).build(frame=cam_isp, nn=nn.out)
    dets_isp = pipeline.create(DrawDetections).build(frame=cam_isp, nn=norm_isp.output, label_map=nn_classes)

    norm_crop = pipeline.create(NormaliezeBbox).build(frame=cam_crop, nn=nn.out)
    dets_crop = pipeline.create(DrawDetections).build(frame=cam_crop, nn=norm_crop.output, label_map=nn_classes)
    
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")
    
    display_crop = pipeline.create(Display).build(frames=dets_crop.output)
    display_crop.setName("Cropped")
    
    print("Pipeline created.")
    pipeline.run()