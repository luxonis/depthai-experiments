import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from AnnotationNode import AnnotationNode
import cv2
import time

IP = "10.12.121.14"
deviceInfo = dai.DeviceInfo(IP)
device = dai.Device(deviceInfo)
platform = device.getPlatform()
visualizer = dai.RemoteConnection(webSocketPort=8765, httpPort=8082)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    networkNode = pipeline.create(ParsingNeuralNetwork).build(
        cameraNode, dai.NNModelDescription("luxonis/mediapipe-face-landmarker:192x192:0.0.1"), fps=10 # make slug a variable in the general example
        )
    
    visualizer_output = cameraNode.requestOutput((960, 960), dai.ImgFrame.Type.NV12, fps=10) # make shape variable in the general example
    
    imageAnnotatorNode = pipeline.create(AnnotationNode) # this would change between different model types
    networkNode.out.link(imageAnnotatorNode.inputDet)
    
    visualizer.addTopic("Frame", visualizer_output, "images")
    visualizer.addTopic("annotations", imageAnnotatorNode.output, "images")
    
    print("Pipeline created.")
    # cam_output = visualizer_output.createOutputQueue()
    
    # output_queue = networkNode.out.createOutputQueue()
    # passthrough_queue = networkNode.passthrough.createOutputQueue()
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        time.sleep(1/30)
