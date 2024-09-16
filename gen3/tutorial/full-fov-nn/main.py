import depthai as dai

from host_nodes.host_display import Display
from host_nodes.keyboard_reader import KeyboardReader
from host_nodes.host_add_detections import DrawDetections
from host_nodes.host_add_detections import NormaliezeBbox
from host_nodes.host_combine import CombineOutputs


model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline() as pipeline:
   
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(812, 608), type=dai.ImgFrame.Type.BGR888p)

    cam_stretch = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.STRETCH)
    cam_crop = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.CROP)
    cam_letterbox = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.LETTERBOX)

    nn_stretch = pipeline.create(dai.node.DetectionNetwork)
    nn_stretch.setConfidenceThreshold(0.5)
    nn_stretch.setNNArchive(nn_archive)
    cam_stretch.link(nn_stretch.input)
    nn_clases = nn_stretch.getClasses()
    
    nn_crop = pipeline.create(dai.node.DetectionNetwork)
    nn_crop.setConfidenceThreshold(0.5)
    nn_crop.setNNArchive(nn_archive)
    cam_crop.link(nn_crop.input)
    nn_clases = nn_crop.getClasses()
    
    nn_letterbox = pipeline.create(dai.node.DetectionNetwork)
    nn_letterbox.setConfidenceThreshold(0.5)
    nn_letterbox.setNNArchive(nn_archive)
    cam_letterbox.link(nn_letterbox.input)
    nn_clases = nn_letterbox.getClasses()
    
    norm_nn_stretch = pipeline.create(NormaliezeBbox).build(frame=cam_stretch, nn=nn_stretch.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_nn = pipeline.create(DrawDetections).build(frame=cam_stretch, nn=norm_nn_stretch.output, label_map=nn_clases)
    
    norm_nn_crop = pipeline.create(NormaliezeBbox).build(frame=cam_crop, nn=nn_crop.out, manip_mode=dai.ImgResizeMode.CROP)
    dets_nn = pipeline.create(DrawDetections).build(frame=cam_crop, nn=norm_nn_crop.output, label_map=nn_clases)
    
    norm_nn_letterbox = pipeline.create(NormaliezeBbox).build(frame=cam_letterbox, nn=nn_letterbox.out, manip_mode=dai.ImgResizeMode.LETTERBOX)
    dets_nn = pipeline.create(DrawDetections).build(frame=cam_letterbox, nn=norm_nn_letterbox.output, label_map=nn_clases)
        

    norm_isp = pipeline.create(NormaliezeBbox).build(frame=cam_isp, nn=nn_stretch.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_isp = pipeline.create(DrawDetections).build(frame=cam_isp, nn=norm_isp.output, label_map=nn_clases)

    keyboard_reader = pipeline.create(KeyboardReader).build(output=norm_isp.output)
    
    display_isp = pipeline.create(Display).build(frames=dets_isp.output)
    display_isp.setName("ISP")

    combined_frames = pipeline.create(CombineOutputs).build(
        nn_manip=dets_nn.output, 
        crop_manip=cam_crop, 
        letterbox_manip=cam_letterbox, 
        stretch_manip=cam_stretch, 
        keyboard_input=keyboard_reader.output
    )
    
    display_combined = pipeline.create(Display).build(frames=combined_frames.output)
    display_combined.setName("Combined")
    
    print("Pipeline created.")
    pipeline.run()