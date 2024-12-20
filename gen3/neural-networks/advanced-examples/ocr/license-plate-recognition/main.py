import argparse
from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.host_process_detections import ProcessDetections
from utils.host_sync import DetectionsRecognitionsSync
from utils.arguments import initialize_argparser
from pathlib import Path

import cv2
import numpy as np

_, args = initialize_argparser()

# visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device( dai.DeviceInfo(args.device) if args.device else dai.DeviceInfo())
platform = device.getPlatform()

FPS = 5
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 20
else:
    frame_type = dai.ImgFrame.Type.BGR888p

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
        input_node = camera_node.requestOutput((1920, 1080), frame_type, fps= FPS)   
    
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
    vehicle_crop_node.inputConfig.setReusePreviousMessage(False)
    vehicle_crop_node.inputImage.setReusePreviousMessage(False)
    vehicle_crop_node.setMaxOutputFrameSize(640*640*3)
    
    config_sender_node.outputs["output_config"].link(vehicle_crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(vehicle_crop_node.inputImage)
    
    # per vehicle license plate detection
    license_plate_detection = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_crop_node.out, "license-plate-detection:640x640"
        )
    
    license_plate_config_sender = pipeline.create(dai.node.Script)
    license_plate_config_sender.setScriptPath(Path(__file__).parent / "utils/license_plate_sender_script.py")

    input_node.link(license_plate_config_sender.inputs['frame_input'])
    config_sender_node.outputs["output_vehicle_detections"].link(license_plate_config_sender.inputs['detections_input'])
    license_plate_detection.out.link(license_plate_config_sender.inputs['license_plate_detections'])
    
    lp_crop_node = pipeline.create(dai.node.ImageManipV2)
    lp_crop_node.inputConfig.setReusePreviousMessage(False)
    lp_crop_node.inputImage.setReusePreviousMessage(False)
    lp_crop_node.setMaxOutputFrameSize(320*48*3)
    
    license_plate_config_sender.outputs["output_config"].link(lp_crop_node.inputConfig)
    license_plate_config_sender.outputs["output_frame"].link(lp_crop_node.inputImage)
    
    # OCR
    ocr_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        lp_crop_node.out, "luxonis/paddle-text-recognition:320x48:cb55eb4"
        )
    ocr_node.getParser(0).setIgnoredIndexes([0, 11, 12, 13, 14, 15, 16, 17, 44, 45, 46, 47, 48, 49, 76, 77, 78, 79, 
                                80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96])

    
    vehicle_lp_queue = license_plate_detection.out.createOutputQueue()
    vehicle_lp_passthrough =  license_plate_detection.passthrough.createOutputQueue()
    vehicle_det_queue = license_plate_config_sender.outputs["output_valid_detections"].createOutputQueue()
    vehicle_det_passthrough = vehicle_detection_node.passthrough.createOutputQueue()
    ocr_queue = ocr_node.out.createOutputQueue()
    
    all_detections_queue = vehicle_detection_node.out.createOutputQueue()
    crop_detections_queue = license_plate_config_sender.outputs["output_valid_crops"].createOutputQueue()
    crop_queue = lp_crop_node.out.createOutputQueue()



    
    # ocr_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
    #     cropNode.out, "paddle-text-recognition:320x48"
    #     )
    # ocr_node.input.setBlocking(True)
    # ocr_node.input.setMaxSize(20)
    
    # sync_node = pipeline.create(DetectionsRecognitionsSync)
    # sync_node.recognitions_input.setBlocking(True)
    # sync_node.recognitions_input.setMaxSize(20)
    
    # ocr_node.out.link(sync_node.recognitions_input)
    # detection_node.passthrough.link(sync_node.passthrough_input)
    # detection_node.out.link(sync_node.detections_input)
    
    # annotation_node = pipeline.create(OCRAnnotationNode)
    # sync_node.out.link(annotation_node.input)
    
    # visualizer.addTopic("Video",resize_node.out )
    # visualizer.addTopic("OCR", annotation_node.white_frame_output)
    # visualizer.addTopic("Text", annotation_node.text_annotations_output)
    
    print("Pipeline created.")
    pipeline.start()
    # visualizer.registerPipeline(pipeline)
    
    crop_index = 0
    while pipeline.isRunning():
        # key = visualizer.waitKey(1)
        # if key == ord('q'):
        #     print("Got q key. Exiting...")
        #     break
        frame = vehicle_det_passthrough.get().getCvFrame()
        detections = vehicle_det_queue.get()
        det_ts = detections.getTimestamp()
        detections = detections.detections
        crop_detections = crop_detections_queue.get().detections
        
        # cropped_frame = vehicle_lp_passthrough.tryGet()
        # license_plate_detections = vehicle_lp_queue.tryGet()
        i = 0
        for detection, lp_detection in zip(detections, crop_detections):
            
            x_min, y_min, x_max, y_max = detection.xmin, detection.ymin, detection.xmax, detection.ymax
            
            x_min = int(x_min * frame.shape[1])
            y_min = int(y_min * frame.shape[0])
            x_max = int(x_max * frame.shape[1])
            y_max = int(y_max * frame.shape[0])
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)        

            w = x_max - x_min
            h = y_max - y_min
            
            ocr_message = ocr_queue.get()
            
            lp_x_min, lp_y_min, lp_x_max, lp_y_max = lp_detection.xmin, lp_detection.ymin, lp_detection.xmax, lp_detection.ymax
            lp_x_min = int(lp_x_min * w) + x_min
            lp_y_min = int(lp_y_min * h) + y_min
            lp_x_max = int(lp_x_max * w) + x_min
            lp_y_max = int(lp_y_max * h) + y_min
            
            lp_x_min = np.clip(lp_x_min, 0, frame.shape[1])
            lp_y_min = np.clip(lp_y_min, 0, frame.shape[0])
            lp_x_max = np.clip(lp_x_max, 0, frame.shape[1])
            lp_y_max = np.clip(lp_y_max, 0, frame.shape[0])
            
            text = " ".join(ocr_message.classes)
            if len(text) < 4:
                continue
            cv2.rectangle(frame, (lp_x_min, lp_y_min), (lp_x_max, lp_y_max), (0, 0, 255), 2)
            cv2.putText(frame, text, (lp_x_min, lp_y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            crop_frame = crop_queue.get().getCvFrame()
            cv2.imshow(f"License Plate {i}", crop_frame)
            
            # cv2.imwrite(f"cropped_frames/license_plate_{crop_index}.jpg", crop_frame)
            # crop_index += 1
            
            i = i + 1
            
        all_detections = all_detections_queue.get().detections
        
        # for detection in all_detections:
        #     x_min, y_min, x_max, y_max = detection.xmin, detection.ymin, detection.xmax, detection.ymax
            
        #     x_min = int(x_min * frame.shape[1])
        #     y_min = int(y_min * frame.shape[0])
        #     x_max = int(x_max * frame.shape[1])
        #     y_max = int(y_max * frame.shape[0])
            
        #     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        #     cv2.putText(frame, str(detection.label) , (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # if cropped_frame is not None and license_plate_detections is not None:
        #     print(cropped_frame.getTransformation())
        #     cropped_frame = cropped_frame.getCvFrame()
        #     for i, detection in enumerate(license_plate_detections.detections):
        #         original_detection = detections[i]
        #         original_w = original_detection.xmax - original_detection.xmin
        #         original_h = original_detection.ymax - original_detection.ymin
                
        #         scale_w = 640 / original_w
        #         scale_h = 640 / original_h
                
                
        #         x_min, y_min, x_max, y_max = detection.xmin, detection.ymin, detection.xmax, detection.ymax
        #         x_min = int(x_min * 640)
        #         y_min = int(y_min * 640)
        #         x_max = int(x_max * 640)
        #         y_max = int(y_max * 640)
        #         cv2.rectangle(cropped_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #         cv2.imshow("License Plate Detection " + str(i), cropped_frame)
        
        cv2.imshow("Vehicle Detection", frame)
        cv2.waitKey(1)
        




# with dai.Pipeline() as pipeline:

#     print("Creating pipeline...")
#     if args.video:
#         replay = pipeline.create(dai.node.ReplayVideo)
#         replay.setReplayVideoFile(Path(args.video).resolve().absolute())
#         replay.setSize(512, 288)
#         replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
#         replay.setFps(FPS)

#         preview = replay.out
#         shaves = 7

#     else:
#         cam = pipeline.create(dai.node.ColorCamera)
#         cam.setPreviewSize(512, 288)
#         cam.setInterleaved(False)
#         cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#         cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#         cam.setFps(FPS)

#         preview = cam.preview
#         shaves = 6

#     to_nn_manip = pipeline.create(dai.node.ImageManip)
#     to_nn_manip.initialConfig.setResize(640, 640)
#     to_nn_manip.initialConfig.setKeepAspectRatio(False)
#     to_nn_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
#     to_nn_manip.setMaxOutputFrameSize(640*640*3)
#     preview.link(to_nn_manip.inputImage)

#     plate_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
#     plate_detection_nn.setConfidenceThreshold(0.5)
#     # plate_detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106"
#                                                         #   , shaves=shaves, version="2021.4"))
#     plate_detection_nn.setNNArchive(plate_detection_nn_archive)

#     plate_detection_nn.input.setBlocking(False)
#     to_nn_manip.out.link(plate_detection_nn.input)

#     car_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
#     car_detection_nn.setConfidenceThreshold(0.5)
#     # car_detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-adas-0002"
#                                                         # , shaves=shaves, version="2021.4"))
#     car_detection_nn.setNNArchive(car_detection_nn_archive)
#     car_detection_nn.input.setBlocking(False)
#     preview.link(car_detection_nn.input)

#     script_plate = pipeline.create(dai.node.Script)
#     preview.link(script_plate.inputs["preview"])
#     plate_detection_nn.out.link(script_plate.inputs["detections"])
#     script_plate.setScript("""
# while True:
#     frame = node.io["preview"].get()
#     detections = node.io["detections"].get().detections
#     license_detections = [detection for detection in detections if detection.label == 2]
    
#     for idx, detection in enumerate(license_detections):
#         cfg = ImageManipConfig()
#         cfg.setKeepAspectRatio(False)
#         cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
#         cfg.setFrameType(ImgFrame.Type.BGR888p)
#         cfg.setResize(94, 24)
        
#         # Send outputs to neural network
#         if idx == 0:
#             node.io["passthrough"].send(frame)
#             cfg.setReusePreviousImage(False)
#         else:
#             cfg.setReusePreviousImage(True)
#         node.io["config"].send(cfg)
#     """)

#     script_car = pipeline.create(dai.node.Script)
#     preview.link(script_car.inputs["preview"])
#     car_detection_nn.out.link(script_car.inputs["detections"])
#     script_car.setScript("""
# while True:
#     frame = node.io["preview"].get()
#     detections = node.io["detections"].get().detections

#     for idx, detection in enumerate(detections):
#         cfg = ImageManipConfig()
#         cfg.setKeepAspectRatio(False)
#         cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
#         cfg.setFrameType(ImgFrame.Type.BGR888p)
#         cfg.setResize(72, 72)

#         # Send outputs to neural network
#         if idx == 0:
#             node.io["passthrough"].send(frame)
#             cfg.setReusePreviousImage(False)
#         else:
#             cfg.setReusePreviousImage(True)
#         node.io["config"].send(cfg)
#     """)

#     manip_plate = pipeline.create(dai.node.ImageManip)
#     manip_plate.inputConfig.setWaitForMessage(True)
#     manip_plate.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
#     manip_plate.initialConfig.setResize(94, 24)
#     script_plate.outputs["passthrough"].link(manip_plate.inputImage)
#     script_plate.outputs["config"].link(manip_plate.inputConfig)

#     manip_car = pipeline.create(dai.node.ImageManip)
#     manip_car.inputConfig.setWaitForMessage(True)
#     manip_car.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
#     manip_car.initialConfig.setResize(72, 72)
#     script_car.outputs["passthrough"].link(manip_car.inputImage)
#     script_car.outputs["config"].link(manip_car.inputConfig)

#     plate_recognition_nn = pipeline.create(dai.node.NeuralNetwork)
#     # plate_recognition_nn.setBlobPath(blobconverter.from_zoo(name="license-plate-recognition-barrier-0007"
#                                                             # , shaves=shaves, version="2021.4"))
#     plate_recognition_nn.setNNArchive(plate_detection_nn_archive)
#     manip_plate.out.link(plate_recognition_nn.input)

#     car_attribute_nn = pipeline.create(dai.node.NeuralNetwork)
#     # car_attribute_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039"
#                                                         # , shaves=shaves, version="2021.4"))
#     car_attribute_nn.setNNArchive(car_attribute_nn_archive)
#     manip_car.out.link(car_attribute_nn.input)

#     plate_manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
#     plate_detection_nn.out.link(plate_manip_sync.input_detections)
#     manip_plate.out.link(plate_manip_sync.input_recognitions)

#     car_manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
#     car_detection_nn.out.link(car_manip_sync.input_detections)
#     manip_car.out.link(car_manip_sync.input_recognitions)

#     plate_recognition_sync = pipeline.create(DetectionsRecognitionsSync).build()
#     plate_detection_nn.out.link(plate_recognition_sync.input_detections)
#     plate_recognition_nn.out.link(plate_recognition_sync.input_recognitions)

#     car_attribute_sync = pipeline.create(DetectionsRecognitionsSync).build()
#     car_detection_nn.out.link(car_attribute_sync.input_detections)
#     car_attribute_nn.out.link(car_attribute_sync.input_recognitions)

#     license_plate_recognition = pipeline.create(LicensePlateRecognition).build(
#         preview=preview,
#         plate_images=plate_manip_sync.output,
#         car_images=car_manip_sync.output,
#         plate_recognitions=plate_recognition_sync.output,
#         car_attributes=car_attribute_sync.output
#     )
#     license_plate_recognition.inputs["preview"].setBlocking(False)
#     license_plate_recognition.inputs["preview"].setMaxSize(16)
#     license_plate_recognition.inputs["plate_images"].setBlocking(False)
#     license_plate_recognition.inputs["car_images"].setBlocking(False)
#     license_plate_recognition.inputs["plate_recognitions"].setBlocking(False)
#     license_plate_recognition.inputs["car_attributes"].setBlocking(False)

#     print("Pipeline created.")
#     pipeline.run()
