import depthai as dai

from host_nodes.host_add_detections import AddDetections
from host_nodes.host_display import Display

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath=archive_path)

with dai.Pipeline() as pipeline:

    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=(812, 608), type=dai.ImgFrame.Type.BGR888p, resizeMode=dai.ImgResizeMode.CROP)
    
    # Crop video to match aspect ratio of the detection network (1:1)
    crop_square = pipeline.create(dai.node.ImageManip)
    crop_square.initialConfig.setResize(720, 720)
    crop_square.setMaxOutputFrameSize(720*720*3)
    preview.link(crop_square.inputImage)

    # Crop video to match detection network
    crop_nn = pipeline.create(dai.node.ImageManip)
    crop_nn.initialConfig.setResize(300, 300)
    preview.link(crop_nn.inputImage)
    
    nn = pipeline.create(dai.node.DetectionNetwork).build(input=crop_nn.out, nnArchive=nn_archive)
    label_map = nn.getClasses()
    
    color = pipeline.create(Display).build(frame=preview)
    color.setName("Color")
      
    full_dets = pipeline.create(AddDetections).build(frame=preview, nn=nn.out, label_map=label_map)
    full = pipeline.create(Display).build(frame=full_dets.output)
    full.setName("Full")
    
    square_dets = pipeline.create(AddDetections).build(frame=crop_square.out, nn=nn.out, label_map=label_map)
    square = pipeline.create(Display).build(frame=square_dets.output)
    square.setName("Square")
    
    passthrough_dets = pipeline.create(AddDetections).build(frame=crop_nn.out, nn=nn.out, label_map=label_map)
    passthrough = pipeline.create(Display).build(frame=passthrough_dets.output)
    passthrough.setName("Passthrough")
       
    print("Pipeline created.")
    pipeline.run()
