import depthai as dai
import argparse
from depthai_nodes import ParsingNeuralNetwork
import cv2
import numpy as np

from host_process_detections import ProcessDetections
from host_sync import CustomSyncNode

FPS = 20

parser = argparse.ArgumentParser()

device = dai.Device()
platform = device.getPlatform()
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
else:
    frame_type = dai.ImgFrame.Type.BGR888p

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    
    cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cameraOutput = cameraNode.requestOutput((1728, 960), frame_type, fps= FPS)
    
    resizeNode = pipeline.create(dai.node.ImageManipV2) 
    resizeNode.initialConfig.setOutputSize(576, 320)
    resizeNode.initialConfig.setReusePreviousImage(False)
    cameraOutput.link(resizeNode.inputImage)
    
    detectionNode: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resizeNode.out, "luxonis/paddle-text-detection:320x576"
        )
    # detectionNode.input.setBlocking(True)
    detectionNode.input.setMaxSize(100)
    
    detectionProcessNode = pipeline.create(ProcessDetections)
    detectionNode.out.link(detectionProcessNode.detections_input)
    
    
    frameSenderNode = pipeline.create(dai.node.Script)
    frameSenderNode.setScript("""
        try:
            while True:
                frame = node.inputs['frame_input'].get()
                num_frames = node.inputs['num_frames_input'].get().getData()[0]
                
                for i in range(num_frames):
                    node.outputs['output_frame'].send(frame)
                
        except Exception as e:
            node.warn(str(e))
    """)
    
    cameraOutput.link(frameSenderNode.inputs['frame_input'])
    detectionProcessNode.num_frames_output.link(frameSenderNode.inputs['num_frames_input'])
    
    cropNode = pipeline.create(dai.node.ImageManipV2)
    cropNode.inputConfig.setMaxSize(100)
    cropNode.inputConfig.setReusePreviousMessage(False)
    cropNode.inputImage.setReusePreviousMessage(False)
    cropNode.inputImage.setMaxSize(100)
    # cropNode.setMaxOutputFrameSize(4976640)
    
    detectionProcessNode.crop_config.link(cropNode.inputConfig)
    frameSenderNode.outputs['output_frame'].link(cropNode.inputImage)
    
    
    ocrNode: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        cropNode.out, "luxonis/paddle-text-recognition:320x48"
        )
    ocrNode.input.setBlocking(True)
    ocrNode.input.setMaxSize(100)
    
    syncNode = pipeline.create(CustomSyncNode)
    syncNode.ocr_inputs.setBlocking(True)
    syncNode.ocr_inputs.setMaxSize(100)
    syncNode.detections_inputs.setMaxSize(100)
    syncNode.passthrough_input.setMaxSize(100)
    
    ocrNode.out.link(syncNode.ocr_inputs)
    detectionNode.passthrough.link(syncNode.passthrough_input)
    detectionNode.out.link(syncNode.detections_inputs)
    
    sync_queue = syncNode.out.createOutputQueue()

    print("Pipeline created.")
    pipeline.start()
    
    while pipeline.isRunning():
        
        message_group = sync_queue.get()
        frame_msg = message_group.__getitem__("passthrough")
        det_msg = message_group.__getitem__("detections")
        ocrs_msg = message_group.__getitem__("ocrs")
        
        frame = frame_msg.getCvFrame()
        white_frame = np.ones(frame.shape, dtype=np.uint8) * 255
        
        for i, det in enumerate(det_msg.detections):
            rect = det.rotated_rect
            points = rect.getPoints()
            points = [[int(p.x * frame.shape[1]), int(p.y * frame.shape[0])] for p in points]
            points = np.array(points, dtype=np.int32)
            location = (points[0] + points[3]) // 2
            text = ocrs_msg.classes[i]
            concatenated_text = "".join(text)
            cv2.putText(white_frame, concatenated_text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        concatenated_frames = np.concatenate((frame, white_frame), axis=1)                
                        
        cv2.imshow("crop", concatenated_frames)
        cv2.waitKey(1)
        
        
                  