#!/usr/bin/env python3

import depthai as dai
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("--webSocketPort", type=int, default=8765)
parser.add_argument("--httpPort", type=int, default=8082)

args = parser.parse_args()

remoteConnector = dai.RemoteConnection(webSocketPort=args.webSocketPort, httpPort=args.httpPort)
ENCODER_PROFILE = dai.VideoEncoderProperties.Profile.H264_MAIN

class ImgAnnotationsGenerator(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.inputDet = self.createInput()
        self.output = self.createOutput()
        self.frame = self.createInput()
        
    def setLabelMap(self, labelMap):
        self.labelMap = labelMap
        
    def run(self):
        while self.isRunning():
            nnData: dai.NNData = self.inputDet.get()
            
            frame = self.frame.get()
            frame_transformation = frame.getTransformation()
            
            detections_transformation = nnData.getTransformation()
            
            # shape = nnData.getTransformation().getSize()
            # print(shape) 
            
            detections = nnData.detections
            imgAnnt = dai.ImgAnnotations()
            imgAnnt.setTimestamp(nnData.getTimestamp())
            annotation = dai.ImgAnnotation()
            
            for detection in detections:
                points = np.array([[detection.xmin, detection.ymin], 
                          [detection.xmax, detection.ymin], 
                          [detection.xmax, detection.ymax], 
                          [detection.xmin, detection.ymax]])
                # points[0, :] *= self.target_shape[0]
                # points[1, :] *= self.target_shape[1]
                points = [dai.Point2f(point[0], point[1]) for point in points]
                
                
                remaped_points = [detections_transformation.remapPointTo(frame_transformation, point) for point in points]
                # print(remaped_points)
                
                pointsAnnotation = dai.PointsAnnotation()
                pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
                pointsAnnotation.points = dai.VectorPoint2f(points)
                # pointsAnnotation.points = dai.VectorPoint2f(remaped_points)
                
                outlineColor = dai.Color(1.0, 0.5, 0.5, 1.0)
                pointsAnnotation.outlineColor = outlineColor
                # fillColor = dai.Color(0.5, 1.0, 0.5, 0.5)
                # pointsAnnotation.fillColor = fillColor
                pointsAnnotation.thickness = 2.0
                text = dai.TextAnnotation()
                text.position = dai.Point2f(detection.xmin, detection.ymin)
                text.text = f"{self.labelMap[detection.label]} {int(detection.confidence * 100)}%"
                text.fontSize = 50.5
                textColor = dai.Color(0.5, 0.5, 1.0, 1.0)
                text.textColor = textColor
                backgroundColor = dai.Color(1.0, 1.0, 0.5, 1.0)
                text.backgroundColor = backgroundColor
                annotation.points.append(pointsAnnotation)
                annotation.texts.append(text)

            imgAnnt.annotations.append(annotation)
            self.output.send(imgAnnt)
            
IP = "10.12.121.230"
deviceInfo = dai.DeviceInfo(IP)
device = dai.Device(deviceInfo)
# device = dai.Device()

# Create pipeline
with dai.Pipeline(device) as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(
        cameraNode, dai.NNModelDescription("yolov6-nano"), fps= 5
    )
    outputToEncode = cameraNode.requestOutput((1920, 1440), type=dai.ImgFrame.Type.NV12, fps=5)
    
    imageAnnotationsGenerator = pipeline.create(ImgAnnotationsGenerator)
    outputToEncode.link(imageAnnotationsGenerator.frame)
    detectionNetwork.out.link(imageAnnotationsGenerator.inputDet)
    
    labelMap = detectionNetwork.getClasses()
    imageAnnotationsGenerator.setLabelMap(labelMap)

    # Add the remote connector topics
    remoteConnector.addTopic("encoded", outputToEncode, "images")
    remoteConnector.addTopic("annotations", imageAnnotationsGenerator.output, "images")
    # remoteConnector.addTopic("preview", detectionNetwork.passthrough, "preview")

    pipeline.start()

    # Register the pipeline with the remote connector
    remoteConnector.registerPipeline(pipeline)

    while pipeline.isRunning():
        if remoteConnector.waitKey(1) == ord("q"):
            pipeline.stop()
            break