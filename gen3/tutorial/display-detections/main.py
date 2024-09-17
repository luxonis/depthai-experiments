import depthai as dai

from host_nodes.host_add_detections import NormaliezeBbox, DrawDetections
from host_nodes.host_display import Display

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath=archive_path)

with dai.Pipeline() as pipeline:

    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=(812, 608), type=dai.ImgFrame.Type.BGR888p)
    square = cam_rgb.requestOutput(size=(720, 720), type=dai.ImgFrame.Type.BGR888p)
    cam_nn = cam_rgb.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p)
    
    nn = pipeline.create(dai.node.DetectionNetwork).build(input=cam_nn, nnArchive=nn_archive)
    label_map = nn.getClasses()
    
    color = pipeline.create(Display).build(frame=preview)
    color.setName("Color")
      
    norm_full = pipeline.create(NormaliezeBbox).build(frame=preview, nn=nn.out)
    full_dets = pipeline.create(DrawDetections).build(frame=preview, nn=norm_full.output, label_map=label_map)
    full = pipeline.create(Display).build(frame=full_dets.output)
    full.setName("Full")
    
    norm_square = pipeline.create(NormaliezeBbox).build(frame=square, nn=nn.out)
    square_dets = pipeline.create(DrawDetections).build(frame=square, nn=norm_square.output, label_map=label_map)
    square = pipeline.create(Display).build(frame=square_dets.output)
    square.setName("Square")
    
    norm_passthrough = pipeline.create(NormaliezeBbox).build(frame=cam_nn, nn=nn.out)
    passthrough_dets = pipeline.create(DrawDetections).build(frame=cam_nn, nn=norm_passthrough.output, label_map=label_map)
    passthrough = pipeline.create(Display).build(frame=passthrough_dets.output)
    passthrough.setName("Passthrough")
       
    print("Pipeline created.")
    pipeline.run()