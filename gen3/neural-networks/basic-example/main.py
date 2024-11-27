import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
import time
from utils.arguments import initialize_argparser

arg_parser, args = initialize_argparser()

visualizer = dai.RemoteConnection()

deviceInfo = dai.DeviceInfo(args.ip)
device = dai.Device(deviceInfo)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    networkNode = pipeline.create(ParsingNeuralNetwork).build(
        cameraNode, dai.NNModelDescription(args.model_slug), fps=args.fps_limit
        )
    
    visualizer.addTopic("Video", networkNode.passthrough, "images") 
    visualizer.addTopic("Visualizations", networkNode.out, "images")
    
    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        time.sleep(1/30)