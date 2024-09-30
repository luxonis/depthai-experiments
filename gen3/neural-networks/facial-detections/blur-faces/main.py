import blobconverter
import depthai as dai
from host_node.blur_bboxes import BlurBboxes
from host_node.host_display import Display
from host_node.normalize_bbox import NormalizeBbox

face_det_model_description = dai.NNModelDescription(
    modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640"
)
face_det_archive_path = dai.getModelFromZoo(face_det_model_description)
face_det_nn_archive = dai.NNArchive(face_det_archive_path)

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    # ColorCamera
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1080, 1080)
    cam.setInterleaved(False)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(
        blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6)
    )
    # face_det_nn.setNNArchive(face_det_nn_archive) #TODO: swap in, when the HostNode->ObjectTracker is ready
    # Link Face ImageManip -> Face detection NN node
    cam.preview.link(face_det_nn.input)

    bbox_norm = pipeline.create(NormalizeBbox).build(
        frame=cam.video, nn=face_det_nn.out, manip_mode=dai.ImgResizeMode.CROP
    )
    blur_faces = pipeline.create(BlurBboxes).build(
        frame=cam.video, nn=bbox_norm.output, rounded_blur=True
    )
    display = pipeline.create(Display).build(frames=blur_faces.output)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline ended.")
