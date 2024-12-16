import depthai as dai
import argparse
from depthai_nodes import ParsingNeuralNetwork
import cv2
import numpy as np
from host_process_detections import ProcessDetections
from host_sync import DetectionsRecognitionsSync

FPS = 5

parser = argparse.ArgumentParser()

device = dai.Device()
platform = device.getPlatform()
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 20
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
    
    detectionProcessNode = pipeline.create(ProcessDetections)
    detectionNode.out.link(detectionProcessNode.detections_input)
    
    
    frameSenderNode = pipeline.create(dai.node.Script)
    frameSenderNode.setScript("""
        try:
            while True:
                frame = node.inputs['frame_input'].get()
                configs_message = node.inputs['config_input'].get()
                
                while configs_message.getTimestamp() > frame.getTimestamp():
                    frame = node.inputs['frame_input'].get() 
                
                for i, cfg in configs_message:
                    node.outputs['output_config'].send(cfg)
                    node.outputs['output_frame'].send(frame)
        except Exception as e:
            node.warn(str(e))
    """)
    
    cameraOutput.link(frameSenderNode.inputs['frame_input'])
    detectionProcessNode.config_output.link(frameSenderNode.inputs['config_input'])
    
    cropNode = pipeline.create(dai.node.ImageManipV2)
    cropNode.inputConfig.setReusePreviousMessage(False)
    cropNode.inputConfig.setMaxSize(20)
    cropNode.inputImage.setReusePreviousMessage(False)
    cropNode.inputImage.setMaxSize(20)
    
    frameSenderNode.outputs["output_config"].link(cropNode.inputConfig)
    frameSenderNode.outputs['output_frame'].link(cropNode.inputImage)
    
    
    ocrNode: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        cropNode.out, "luxonis/paddle-text-recognition:320x48"
        )
    ocrNode.input.setBlocking(True)
    ocrNode.input.setMaxSize(20)
    
    syncNode = pipeline.create(DetectionsRecognitionsSync)
    syncNode.recognitions_input.setBlocking(True)
    syncNode.recognitions_input.setMaxSize(20)
    
    ocrNode.out.link(syncNode.recognitions_input)
    detectionNode.passthrough.link(syncNode.passthrough_input)
    detectionNode.out.link(syncNode.detections_input)
    
    sync_queue = syncNode.out.createOutputQueue()
    
    print("Pipeline created.")
    pipeline.start()
    
    while pipeline.isRunning():
        
        message_group = sync_queue.get()
        frame_msg = message_group["passthrough"]
        det_msg = message_group["detections"]
        ocrs_msg = message_group["recognitions"]
        
        frame = frame_msg.getCvFrame()
        white_frame = np.ones(frame.shape, dtype=np.uint8) * 255
        
        for i, det in enumerate(det_msg.detections):
            rect = det.rotated_rect
            points = rect.getPoints()
            points = [[int(p.x * frame.shape[1]), int(p.y * frame.shape[0])] for p in points]
            points = np.array(points, dtype=np.int32)
            location = (points[0] + points[3]) // 2
            text = ocrs_msg.recognitions[i].classes
            concatenated_text = "".join(text)
            cv2.putText(white_frame, concatenated_text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        concatenated_frames = np.concatenate((frame, white_frame), axis=1)                
                        
        cv2.imshow("crop", concatenated_frames)
        cv2.waitKey(1)
        
        
                  