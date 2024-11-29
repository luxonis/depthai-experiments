import depthai as dai
import argparse
from depthai_nodes import ParsingNeuralNetwork
import time
import cv2
import numpy as np

from host_process_detections import ProcessDetections
from host_sync import CustomSyncNode

FPS = 10

parser = argparse.ArgumentParser()
parser.add_argument("--ip", 
                    help="Specify the IP address of your RVC4 device",
                    default="10.12.121.123",
                    )

deviceInfo = dai.DeviceInfo("10.12.121.85")
device = dai.Device(deviceInfo)
platform = device.getPlatform()

detection_model_description = dai.NNModelDescription("luxonis/paddle-text-detection:320x576", platform="RVC4")
detection_nn_archive = dai.NNArchive(dai.getModelFromZoo(detection_model_description))

ocr_model_description = dai.NNModelDescription("luxonis/paddle-text-recognition:320x48", platform="RVC4")
ocr_nn_archive = dai.NNArchive(dai.getModelFromZoo(ocr_model_description))


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cameraOutput = cameraNode.requestOutput((1728, 960), dai.ImgFrame.Type.BGR888i, fps= FPS)
    
    resizeNode = pipeline.create(dai.node.ImageManipV2)
    resizeNode.initialConfig.addResize(576, 320)
    cameraOutput.link(resizeNode.inputImage)
    
    detectionNode = pipeline.create(ParsingNeuralNetwork).build(
        resizeNode.out, detection_nn_archive
        )
    
    detectionProcessNode = pipeline.create(ProcessDetections)
    detectionNode.out.link(detectionProcessNode.detections_input)
    cameraOutput.link(detectionProcessNode.frame)
    
    
    cropNode = pipeline.create(dai.node.ImageManipV2)
    detectionProcessNode.crop_config.link(cropNode.inputConfig)
    detectionProcessNode.output_frame.link(cropNode.inputImage)

    # ocrNode = pipeline.create(ParsingNeuralNetwork).build(
    #     cropNode.out, ocr_nn_archive
    #     )
    
    
    # syncNode = pipeline.create(CustomSyncNode)
    # ocrNode.out.link(syncNode.ocr_inputs)
    # detectionNode.passthrough.link(syncNode.passthrough_input)
    # detectionNode.out.link(syncNode.detections_inputs)
    
    cropQueue = cropNode.out.createOutputQueue()
    # out_queue = syncNode.output.createOutputQueue()
    # det_queue = detectionNode.out.createOutputQueue()
    print("Pipeline created.")
    pipeline.start()
    
    while pipeline.isRunning():
        # outputs = out_queue.get()
        
        # frame = outputs.passthrough.getCvFrame()
        
        # detections = outputs.detections
        
        # for det in detections:
        #     classes = det.classes
        #     rect = det.detection.rotated_rect
            
        #     points = rect.getPoints()
        #     points = [[int(p.x * frame.shape[1]), int(p.y * frame.shape[0])] for p in points]
        #     points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        #     cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        #     for txt in classes:
        #         cv2.putText(frame, txt, points[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
                        
        # cv2.imshow("cvframe", frame)
        # cv2.waitKey(1)
        
        crop = cropQueue.get().getCvFrame()
        
        cv2.imshow("crop", crop)
        cv2.waitKey(1)
        # time.sleep(1/FPS)
        
        
                  