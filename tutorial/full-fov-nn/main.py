import depthai as dai

from host_node.normalize_bbox import NormalizeBbox
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.keyboard_reader import KeyboardReader
from combine_outputs import CombineOutputs

device = dai.Device()
platform = device.getPlatform()

if platform == dai.Platform.RVC4:
    raise RuntimeError("ImgResizeMode.LETTERBOX is not yet supported on RVC4 platform")

model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline(device) as pipeline:

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_isp = cam.requestOutput(size=(812, 608), type=dai.ImgFrame.Type.NV12)

    output_type = dai.ImgFrame.Type.BGR888i if platform == dai.Platform.RVC4 else dai.ImgFrame.Type.BGR888p
    cam_stretch = cam.requestOutput(size=(512, 288), type=output_type, resizeMode=dai.ImgResizeMode.STRETCH)
    cam_crop = cam.requestOutput(size=(512, 288), type=output_type, resizeMode=dai.ImgResizeMode.CROP)
    cam_letterbox = cam.requestOutput(size=(512, 288), type=output_type, resizeMode=dai.ImgResizeMode.LETTERBOX)

    nn_stretch = pipeline.create(dai.node.DetectionNetwork).build(cam_stretch, nn_archive, confidenceThreshold=0.5)
    nn_clases = nn_stretch.getClasses()
    
    nn_crop = pipeline.create(dai.node.DetectionNetwork).build(cam_crop, nn_archive, confidenceThreshold=0.5)
    nn_clases = nn_crop.getClasses()
    
    nn_letterbox = pipeline.create(dai.node.DetectionNetwork).build(cam_letterbox, nn_archive, confidenceThreshold=0.5)
    nn_clases = nn_letterbox.getClasses()
    
    norm_nn_stretch = pipeline.create(NormalizeBbox).build(frame=cam_stretch, nn=nn_stretch.out, manip_mode=dai.ImgResizeMode.STRETCH)
    dets_nn = pipeline.create(DrawDetections).build(frame=cam_stretch, nn=norm_nn_stretch.output, label_map=nn_clases)
    
    norm_nn_crop = pipeline.create(NormalizeBbox).build(frame=cam_crop, nn=nn_crop.out, manip_mode=dai.ImgResizeMode.CROP)
    dets_nn = pipeline.create(DrawDetections).build(frame=cam_crop, nn=norm_nn_crop.output, label_map=nn_clases)
    
    norm_nn_letterbox = pipeline.create(NormalizeBbox).build(frame=cam_letterbox, nn=nn_letterbox.out, manip_mode=dai.ImgResizeMode.LETTERBOX)
    dets_nn = pipeline.create(DrawDetections).build(frame=cam_letterbox, nn=norm_nn_letterbox.output, label_map=nn_clases)

    norm_isp = pipeline.create(NormalizeBbox).build(frame=cam_isp, nn=nn_stretch.out, manip_mode=dai.ImgResizeMode.STRETCH)
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