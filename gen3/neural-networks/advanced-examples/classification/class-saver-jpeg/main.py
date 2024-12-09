import depthai as dai
from class_saver import ClassSaver
from host_node.visualize_detections import VisualizeDetections

device = dai.Device()
platform = device.getPlatform().name

model_description = dai.NNModelDescription(
    modelSlug="yolov10-nano", platform=platform, modelVersionSlug="coco-512x288"
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

VIDEO_SIZE = (1280, 720)
YOLO_SIZE = (512, 288)

visualizer = dai.RemoteConnection()

# Start defining a pipeline
with dai.Pipeline(device) as pipeline:
    cam_rgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    color_out = cam_rgb.requestOutput(
        size=VIDEO_SIZE, type=dai.ImgFrame.Type.BGR888p, fps=15
    )

    detection_manip = pipeline.create(dai.node.ImageManip)
    detection_manip.initialConfig.setResize(*YOLO_SIZE)
    color_out.link(detection_manip.inputImage)

    detection_nn = pipeline.create(dai.node.DetectionNetwork).build(
        input=detection_manip.out, nnArchive=nn_archive, confidenceThreshold=0.5
    )
    detection_nn.input.setBlocking(False)
    detection_nn.input.setMaxSize(2)

    visualize_detections = pipeline.create(VisualizeDetections).build(
        nn=detection_nn.out, label_map=detection_nn.getClasses()
    )

    class_saver = pipeline.create(ClassSaver).build(
        frames=color_out, nn=detection_nn.out, classes=detection_nn.getClasses()
    )

    visualizer.addTopic("Detections", visualize_detections.output)
    visualizer.addTopic("Color", color_out)

    pipeline.start()
    visualizer.registerPipeline(pipeline)
    print("Pipeline created.")
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
