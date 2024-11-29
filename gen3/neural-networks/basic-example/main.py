import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
import time
from utils.arguments import initialize_argparser

arg_parser, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device_info = dai.DeviceInfo(args.device)
device = dai.Device(device_info)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    camera_node = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        camera_node, dai.NNModelDescription(args.model_slug), fps=args.fps_limit
        )
    
    visualizer.addTopic("Video", camera_node.passthrough, "images") 
    visualizer.addTopic("Visualizations", camera_node.out, "images")
    
    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        time.sleep(1/30)