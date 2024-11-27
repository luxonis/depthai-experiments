import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
import cv2
import numpy as np 

IP = "10.12.121.121"
deviceInfo = dai.DeviceInfo(IP)
device = dai.Device(deviceInfo)
# device = dai.Device()
platform = device.getPlatform()
            
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    networkNode = pipeline.create(ParsingNeuralNetwork).build(
        cameraNode, dai.NNModelDescription("luxonis/paddle-text-detection:320x576"), fps=30 # make slug a variable in the general example
        )
    
    cam_output = cameraNode.requestOutput((640, 640), dai.ImgFrame.Type.BGR888i, fps=30)
    cam_queue = cam_output.createOutputQueue()
    output_queue = networkNode.out.createOutputQueue()
    passthrough_queue = networkNode.passthrough.createOutputQueue()
    pipeline.start()
    
    while pipeline.isRunning():
        frame = passthrough_queue.get().getCvFrame()
        width, height = frame.shape[1], frame.shape[0]
        for detection in output_queue.get().detections:
            points = detection.rotated_rect.getPoints()
            points = [[point.x, point.y] for point in points]
            points = [[int(point[0]*width), int(point[1]*height)] for point in points]
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))  # Convert to required format
            cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        
        cv2.imshow("passthrough", frame)
        cv2.waitKey(1)
