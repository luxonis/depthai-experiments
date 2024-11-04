import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from host_node.host_depth_color_transform import DepthColorTransform
from host_node.overlay_frames import OverlayFrames
from host_node.visualize_detections_v2 import VisualizeDetectionsV2
from yolo_labels import LABEL_MAP

NN_SIZE = (512, 288)

device = dai.Device()

nn_model_description = dai.NNModelDescription(
    modelSlug="yolov8-instance-segmentation-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="coco-512x288",
)

nn_archive_path = dai.getModelFromZoo(nn_model_description, useCached=True)
nn_archive = dai.NNArchive(nn_archive_path)

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = cam.requestOutput(NN_SIZE, dai.ImgFrame.Type.BGR888p, fps=10)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=color_out, nn_source=nn_archive
    )
    visualize_detections = pipeline.create(VisualizeDetectionsV2).build(
        nn.out, label_map=LABEL_MAP
    )

    color_transform = pipeline.create(DepthColorTransform).build(
        visualize_detections.output_mask
    )
    color_transform.setMaxDisparity(len(LABEL_MAP) + 1)

    overlay = pipeline.create(OverlayFrames).build(color_out, color_transform.output)

    print("Pipeline created.")
    visualizer.addTopic("Detections", visualize_detections.output)
    visualizer.addTopic("Color", color_out)
    visualizer.addTopic("Mask", overlay.output)
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
