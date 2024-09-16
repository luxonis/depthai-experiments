import depthai as dai
<<<<<<< HEAD
import cv2
import numpy as np
=======
from host_nodes.host_add_detections import NormaliezeBbox, DrawDetections
from host_nodes.host_display import Display
>>>>>>> 48d54bf (full fov tutorial refactored)

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)


with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.BGR888p, fps=25, resizeMode=dai.ImgResizeMode.STRETCH)
    cam_stretch = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.STRETCH)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    nn_classes = nn.getClasses()

    cam_stretch.link(nn.input)

    norm_isp = pipeline.create(NormaliezeBbox).build(frame=cam_isp, nn=nn.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_isp = pipeline.create(DrawDetections).build(frame=cam_isp, nn=norm_isp.output, label_map=nn_classes)

    norm_stretch = pipeline.create(NormaliezeBbox).build(frame=cam_stretch, nn=nn.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_stretch = pipeline.create(DrawDetections).build(frame=cam_stretch, nn=norm_stretch.output, label_map=nn_classes)
    
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")
    
    display_crop = pipeline.create(Display).build(frames=dets_stretch.output)
    display_crop.setName("Stretched")
    
    print("Pipeline created.")
    pipeline.run()
