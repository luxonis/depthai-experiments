import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from detection_segmentation_node import DetSegAnntotationNode
from utils.arguments import initialize_argparser
import time

arg_parser, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device != "" else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    camera_node = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        camera_node, dai.NNModelDescription(args.model_slug), fps=args.fps_limit
        )
    
    annotation_node = pipeline.create(DetSegAnntotationNode)
    nn_with_parser.passthrough.link(annotation_node.input_frame)
    nn_with_parser.out.link(annotation_node.input_detections)

    visualizer.addTopic("Video", annotation_node.out, "images")
    visualizer.addTopic("Detections", nn_with_parser.out, "detections")
    
    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break