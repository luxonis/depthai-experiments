import depthai as dai

from host_nodes.host_display import Display
from host_nodes.host_add_detections import AddDetections

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)


with dai.Pipeline() as pipeline:
    
    cam = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    label_map = nn.getClasses()
    preview.link(nn.input)

    rgb_dets = pipeline.create(AddDetections).build(frame=preview, nn=nn.out, label_map=label_map)
    rgb = pipeline.create(Display).build(frame=rgb_dets.output)
    rgb.setName("RGB")
    
    print("Pipeline created.")
    pipeline.run()
