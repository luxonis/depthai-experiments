import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

_, args = initialize_argparser()


device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

visualizer = dai.RemoteConnection(httpPort=8082)


class FilterDets(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput()

    def build(self, detections: dai.Node.Output):
        self.link_args(detections)
        return self

    def process(self, detections: dai.ImgDetections):
        new_dets = dai.ImgDetections()
        new_dets_list = []
        for detection in detections.detections:
            if detection.label != 57:
                new_dets_list.append(detection)
        new_dets.detections = new_dets_list
        new_dets.setTimestamp(detections.getTimestamp())
        new_dets.setSequenceNum(detections.getSequenceNum())
        self.output.send(new_dets)


with dai.Pipeline(device) as pipeline:
    platform = device.getPlatform()

    model_description = dai.NNModelDescription(
        "luxonis/yolov6-nano:r2-coco-512x288", platform=platform.name
    )
    archive_path = dai.getModelFromZoo(model_description)
    nn_archive = dai.NNArchive(archivePath=archive_path)

    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput(
        (512, 288),
        fps=args.fps_limit,
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
    )

    nn = pipeline.create(ParsingNeuralNetwork).build(cam_out, nn_archive)

    filter_dets = pipeline.create(FilterDets).build(nn.out)

    visualizer.addTopic("Passthrough", nn.passthrough)
    visualizer.addTopic("Detections", filter_dets.output)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
