import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from seg_annotation_node import SegAnnotationNode
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
    
    annotation_node = pipeline.create(SegAnnotationNode)
    nn_with_parser.passthrough.link(annotation_node.input_frame)
    nn_with_parser.out.link(annotation_node.input_segmentation)

    visualizer.addTopic("Video", annotation_node.out, "images") 
    
    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        time.sleep(1/30)