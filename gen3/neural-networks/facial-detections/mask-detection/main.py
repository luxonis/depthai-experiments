import depthai as dai
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.normalize_detections import NormalizeDetections
from mask_detection import MaskDetection

device = dai.Device()

ppe_model_description = dai.NNModelDescription(
    modelSlug="ppe-detection",
    platform=device.getPlatform().name,
    modelVersionSlug="640x640",
)
ppe_model_archive_path = dai.getModelFromZoo(ppe_model_description)
ppe_nn_archive = dai.NNArchive(ppe_model_archive_path)

NN_INPUT_SIZE = (640, 640)

with dai.Pipeline(device) as pipeline:
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    preview = cam.requestOutput(
        size=NN_INPUT_SIZE, type=dai.ImgFrame.Type.BGR888p, fps=10
    )
    ppe_nn = pipeline.create(dai.node.DetectionNetwork).build(
        input=preview, nnArchive=ppe_nn_archive, confidenceThreshold=0.5
    )
    ppe_nn.setNumInferenceThreads(2)
    mask_detection = pipeline.create(MaskDetection).build(ppe_nn=ppe_nn.out)
    normalize_mask_detections = pipeline.create(NormalizeDetections).build(
        frame=preview, nn=mask_detection.output
    )
    draw_mask_detections = pipeline.create(DrawDetections).build(
        frame=preview,
        nn=normalize_mask_detections.output,
        label_map=mask_detection.LABELS,
    )
    display = pipeline.create(Display).build(frames=draw_mask_detections.output)

    pipeline.run()
