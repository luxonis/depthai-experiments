import depthai as dai
from host_node.visualize_detections import VisualizeDetections
from qr_decoder import QRDecoder

device = dai.Device()

# if model changed change README
model_description = dai.NNModelDescription(
    modelSlug="qrdet",
    modelVersionSlug="nano-288x512",
    platform=device.getPlatform().name,
)
archivePath = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath)

VIDEO_SIZE = (1920, 1080)
DETECTION_NN_SIZE = (512, 288)

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=20)

    manip = pipeline.create(dai.node.ImageManipV2)
    manip.initialConfig.addResize(*DETECTION_NN_SIZE)
    color_out.link(manip.inputImage)

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        nnArchive=nn_archive, input=manip.out, confidenceThreshold=0.3
    )
    nn.input.setMaxSize(1)
    nn.input.setBlocking(False)

    visualize_detections = pipeline.create(VisualizeDetections).build(
        nn=nn.out, label_map=nn.getClasses()
    )
    visualize_detections.set_draw_labels(False)
    visualize_detections.set_draw_confidence(False)

    scanner = pipeline.create(QRDecoder).build(frame=color_out, nn=nn.out)

    visualizer.addTopic("Detections", visualize_detections.output)
    visualizer.addTopic("Decoded text", scanner.output)
    visualizer.addTopic("Color", color_out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
