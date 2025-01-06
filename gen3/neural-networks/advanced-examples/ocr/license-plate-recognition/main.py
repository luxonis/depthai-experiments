import argparse
from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from pathlib import Path

import cv2
import numpy as np

_, args = initialize_argparser()

device = dai.Device( dai.DeviceInfo(args.device) if args.device else dai.DeviceInfo())
platform = device.getPlatform()

FPS = 5
frame_type = dai.ImgFrame.Type.BGR888p
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 15

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    
    if args.media_path:
        replay_node = pipeline.create(dai.node.ReplayVideo)
        replay_node.setReplayVideoFile(Path(args.media_path))
        replay_node.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay_node.setLoop(True)
        
        video_resize_node = pipeline.create(dai.node.ImageManipV2)
        video_resize_node.initialConfig.setOutputSize(1920*2, 1080*2)
        video_resize_node.initialConfig.setFrameType(frame_type)
        video_resize_node.initialConfig.setReusePreviousImage(False)
        video_resize_node.setMaxOutputFrameSize(1920*1080*3 * 4)
        
        replay_node.out.link(video_resize_node.inputImage)
        
        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1920*2, 1080*2), frame_type, fps= FPS)   
    
    vehicle_detection_resize_node = pipeline.create(dai.node.ImageManipV2) 
    vehicle_detection_resize_node.initialConfig.setOutputSize(512, 288)
    vehicle_detection_resize_node.initialConfig.setReusePreviousImage(False)
    input_node.link(vehicle_detection_resize_node.inputImage)
    
    # vehicle detection
    vehicle_detection_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_detection_resize_node.out, "yolov6-nano:r2-coco-512x288"
        )
    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(Path(__file__).parent / "utils/config_sender_script.py")
    
    input_node.link(config_sender_node.inputs['frame_input'])
    vehicle_detection_node.out.link(config_sender_node.inputs['detections_input'])

    vehicle_crop_node = pipeline.create(dai.node.ImageManipV2)
    vehicle_crop_node.initialConfig.setReusePreviousImage(False)
    vehicle_crop_node.inputConfig.setReusePreviousMessage(False)
    vehicle_crop_node.inputImage.setReusePreviousMessage(False)
    vehicle_crop_node.setMaxOutputFrameSize(640*640*3)
    
    config_sender_node.outputs["output_config"].link(vehicle_crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(vehicle_crop_node.inputImage)
    
    # per vehicle license plate detection
    lp_config_sender = pipeline.create(dai.node.Script)
    lp_config_sender.setScriptPath(Path(__file__).parent / "utils/license_plate_sender_script.py")
    input_node.link(lp_config_sender.inputs['frame_input'])

    
    license_plate_detection = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_crop_node.out, "license-plate-detection:640x640"
        )
    config_sender_node.outputs["output_vehicle_detections"].link(lp_config_sender.inputs['detections_input'])
    license_plate_detection.out.link(lp_config_sender.inputs['license_plate_detections'])
    
    lp_crop_node = pipeline.create(dai.node.ImageManipV2)
    vehicle_crop_node.initialConfig.setReusePreviousImage(False)
    lp_crop_node.inputConfig.setReusePreviousMessage(False)
    lp_crop_node.inputImage.setReusePreviousMessage(False)
    lp_crop_node.setMaxOutputFrameSize(320*48*3)
    
    lp_config_sender.outputs["lp_crop_config"].link(lp_crop_node.inputConfig)
    lp_config_sender.outputs["lp_crop_frame"].link(lp_crop_node.inputImage)
    
    # OCR
    ocr_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        lp_crop_node.out, "luxonis/paddle-text-recognition:320x48"
        )
    ocr_node.getParser(0).setIgnoredIndexes([0, 11, 12, 13, 14, 15, 16, 17, 44, 45, 46, 47, 48, 49, 76, 77, 78, 79, 
                                80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96])

    # link to output queues
    vehicle_detections_queue = lp_config_sender.outputs["output_valid_detections"].createOutputQueue()
    vehicle_det_passthrough_image = vehicle_detection_node.passthrough.createOutputQueue()
    ocr_output_queue = ocr_node.out.createOutputQueue()
    lp_crop_detections_queue = lp_config_sender.outputs["output_valid_crops"].createOutputQueue()
    lp_crop_image_queue = lp_crop_node.out.createOutputQueue()
    
    print("Pipeline created.")
    pipeline.start()

    while pipeline.isRunning():
        frame = vehicle_det_passthrough_image.get().getCvFrame()
        frame_h, frame_w = frame.shape[:2]   
        
        detections = vehicle_detections_queue.get().detections
        crop_detections = lp_crop_detections_queue.get().detections
        
        for detection, lp_detection in zip(detections, crop_detections):
            x_min = int(detection.xmin * frame_w)
            y_min = int(detection.ymin * frame_h)
            x_max = int(detection.xmax * frame_w)
            y_max = int(detection.ymax * frame_h)
            
            vehicle_w = x_max - x_min
            vehicle_h = y_max - y_min
            
            ocr_message = ocr_output_queue.get()
            text = "".join(ocr_message.classes)
            license_plate = lp_crop_image_queue.get().getCvFrame()
            
            if len(text) < 5:
                continue
            
            lp_x_min = int(lp_detection.xmin * vehicle_w) + x_min
            lp_y_min = int(lp_detection.ymin * vehicle_h) + y_min
            lp_x_max = int(lp_detection.xmax * vehicle_w) + x_min
            lp_y_max = int(lp_detection.ymax * vehicle_h) + y_min
            
            lp_x_min = np.clip(lp_x_min, 0, frame_w)
            lp_y_min = np.clip(lp_y_min, 0, frame_h)
            lp_x_max = np.clip(lp_x_max, 0, frame_w)
            lp_y_max = np.clip(lp_y_max, 0, frame_h)
            
            license_plate = cv2.resize(license_plate, (80, 12))

            white_frame = np.ones((12, 80, 3)) * 255
            cv2.putText(white_frame, text,(2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            crop_region = frame[lp_y_max:lp_y_max + 24, lp_x_min:lp_x_min+80]
            lp_text = np.concatenate((license_plate, white_frame), axis=0)
            lp_text = cv2.resize(lp_text, (crop_region.shape[1], crop_region.shape[0]))
            
            frame[lp_y_max:lp_y_max + crop_region.shape[0], lp_x_min:lp_x_min+crop_region.shape[1]] = lp_text

        cv2.imshow("License Plate Detection", frame)
        cv2.waitKey(1)
        
    pipeline.stop()
    