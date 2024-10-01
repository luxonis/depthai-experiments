import depthai as dai
from depthai_nodes.ml.parsers import YuNetParser
from host_node.blur_bboxes import BlurBboxes
from host_node.host_display import Display
from host_node.normalize_bbox import NormalizeBbox
from host_node.yunet_bridge import YuNetBridge

device = dai.Device()
face_det_model_description = dai.NNModelDescription(
    modelSlug="yunet", platform=device.getPlatform().name, modelVersionSlug="640x640"
)
face_det_archive_path = dai.getModelFromZoo(face_det_model_description)
face_det_nn_archive = dai.NNArchive(face_det_archive_path)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    # Camera
    print("Creating Camera...")
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    cam_nn = cam.requestOutput(size=(640, 640), type=dai.ImgFrame.Type.BGR888p)
    cam_square = cam.requestOutput(size=(1080, 1080), type=dai.ImgFrame.Type.BGR888p)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.NeuralNetwork)
    face_det_nn.setNNArchive(face_det_nn_archive)

    cam_nn.link(face_det_nn.input)
    parser = pipeline.create(YuNetParser)
    face_det_nn.out.link(parser.input)
    parser.setConfidenceThreshold(0.5)

    bridge = pipeline.create(YuNetBridge).build(nn=parser.out)
    bbox_norm = pipeline.create(NormalizeBbox).build(
        frame=cam_square, nn=bridge.output, manip_mode=dai.ImgResizeMode.CROP
    )
    blur_faces = pipeline.create(BlurBboxes).build(
        frame=cam_square, nn=bbox_norm.output, rounded_blur=True
    )
    display = pipeline.create(Display).build(frames=blur_faces.output)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline ended.")
