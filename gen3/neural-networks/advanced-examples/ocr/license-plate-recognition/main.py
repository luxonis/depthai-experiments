import argparse
from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.host_process_detections import ProcessDetections
from utils.host_sync import DetectionsRecognitionsSync
from utils.arguments import initialize_argparser
from pathlib import Path

_, args = initialize_argparser()

visualizer = dai.RemoteConnection()
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
        video_resize_node.initialConfig.setOutputSize(1920, 1080)
        video_resize_node.initialConfig.setFrameType(frame_type)
        
        replay_node.out.link(video_resize_node.inputImage)
        
        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1920, 1080), frame_type, fps= FPS)   
    
    vehicle_detection_resize_node = pipeline.create(dai.node.ImageManipV2) 
    vehicle_detection_resize_node.initialConfig.setOutputSize(512, 288)
    vehicle_detection_resize_node.initialConfig.setReusePreviousImage(False)
    input_node.link(vehicle_detection_resize_node.inputImage)
    
    
    vehicle_detection_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_detection_resize_node.out, "yolov6-nano:r2-coco-512x288"
        )
    
    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScript("""
        try:
            while True:
                frame = node.inputs['frame_input'].get()
                detections_message = node.inputs['detections_input'].get()
                
                for d in detections_message.detections:
                    det_w = d.xmax - d.xmin
                    det_h = d.ymax - d.ymin
                    det_center = dai.Point2f((d.xmin + d.xmax) / 2, (d.ymin + d.ymax) / 2)
                    det_size = dai.Size2f(det_w, det_h)
                    det_rect = dai.RotatedRect(det_center, det_size, 0)
                    det_rect = det_rect.denormalize(frame.getWidth(), frame.getHeight())
                    
                    cfg = dai.ImageManipConfig()
                    cfg.addCropRotatedRect(det_rect, normalizedCoords=False)
                    cfg.setOutputSize(640, 640)
                    cfg.setReusePreviousImage(False)
                    cfg.setTimestamp(detections_message.getTimestamp())
                    
                    node.outputs['output_config'].send(cfg)
                    node.outputs['output_frame'].send(frame)

        except Exception as e:
            node.warn(str(e))
    """)
    input_node.link(config_sender_node.inputs['frame_input'])
    vehicle_detection_node.out.link(config_sender_node.inputs['detections_input'])
    
    vehicle_crop_node = pipeline.create(dai.node.ImageManipV2)
    vehicle_crop_node.inputConfig.setReusePreviousMessage(False)
    vehicle_crop_node.inputImage.setReusePreviousMessage(False)
    
    config_sender_node.outputs["output_config"].link(vehicle_crop_node.inputConfig)
    config_sender_node.outputs['output_frame'].link(vehicle_crop_node.inputImage)
    
    
    
    
    license_plate_detection = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_crop_node.out, "license-plate-detection:640x640"
        )
    
    ocr_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        cropNode.out, "paddle-text-recognition:320x48"
        )
    ocr_node.input.setBlocking(True)
    ocr_node.input.setMaxSize(20)
    
    sync_node = pipeline.create(DetectionsRecognitionsSync)
    sync_node.recognitions_input.setBlocking(True)
    sync_node.recognitions_input.setMaxSize(20)
    
    ocr_node.out.link(sync_node.recognitions_input)
    detection_node.passthrough.link(sync_node.passthrough_input)
    detection_node.out.link(sync_node.detections_input)
    
    annotation_node = pipeline.create(OCRAnnotationNode)
    sync_node.out.link(annotation_node.input)
    
    visualizer.addTopic("Video",resize_node.out )
    visualizer.addTopic("OCR", annotation_node.white_frame_output)
    visualizer.addTopic("Text", annotation_node.text_annotations_output)
    
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord('q'):
            print("Got q key. Exiting...")
            break







with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(512, 288)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setFps(FPS)

        preview = replay.out
        shaves = 7

    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(512, 288)
        cam.setInterleaved(False)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(FPS)

        preview = cam.preview
        shaves = 6

    to_nn_manip = pipeline.create(dai.node.ImageManip)
    to_nn_manip.initialConfig.setResize(640, 640)
    to_nn_manip.initialConfig.setKeepAspectRatio(False)
    to_nn_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    to_nn_manip.setMaxOutputFrameSize(640*640*3)
    preview.link(to_nn_manip.inputImage)

    plate_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    plate_detection_nn.setConfidenceThreshold(0.5)
    # plate_detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106"
                                                        #   , shaves=shaves, version="2021.4"))
    plate_detection_nn.setNNArchive(plate_detection_nn_archive)

    plate_detection_nn.input.setBlocking(False)
    to_nn_manip.out.link(plate_detection_nn.input)

    car_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    car_detection_nn.setConfidenceThreshold(0.5)
    # car_detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-adas-0002"
                                                        # , shaves=shaves, version="2021.4"))
    car_detection_nn.setNNArchive(car_detection_nn_archive)
    car_detection_nn.input.setBlocking(False)
    preview.link(car_detection_nn.input)

    script_plate = pipeline.create(dai.node.Script)
    preview.link(script_plate.inputs["preview"])
    plate_detection_nn.out.link(script_plate.inputs["detections"])
    script_plate.setScript("""
while True:
    frame = node.io["preview"].get()
    detections = node.io["detections"].get().detections
    license_detections = [detection for detection in detections if detection.label == 2]
    
    for idx, detection in enumerate(license_detections):
        cfg = ImageManipConfig()
        cfg.setKeepAspectRatio(False)
        cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        cfg.setFrameType(ImgFrame.Type.BGR888p)
        cfg.setResize(94, 24)
        
        # Send outputs to neural network
        if idx == 0:
            node.io["passthrough"].send(frame)
            cfg.setReusePreviousImage(False)
        else:
            cfg.setReusePreviousImage(True)
        node.io["config"].send(cfg)
    """)

    script_car = pipeline.create(dai.node.Script)
    preview.link(script_car.inputs["preview"])
    car_detection_nn.out.link(script_car.inputs["detections"])
    script_car.setScript("""
while True:
    frame = node.io["preview"].get()
    detections = node.io["detections"].get().detections

    for idx, detection in enumerate(detections):
        cfg = ImageManipConfig()
        cfg.setKeepAspectRatio(False)
        cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        cfg.setFrameType(ImgFrame.Type.BGR888p)
        cfg.setResize(72, 72)

        # Send outputs to neural network
        if idx == 0:
            node.io["passthrough"].send(frame)
            cfg.setReusePreviousImage(False)
        else:
            cfg.setReusePreviousImage(True)
        node.io["config"].send(cfg)
    """)

    manip_plate = pipeline.create(dai.node.ImageManip)
    manip_plate.inputConfig.setWaitForMessage(True)
    manip_plate.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip_plate.initialConfig.setResize(94, 24)
    script_plate.outputs["passthrough"].link(manip_plate.inputImage)
    script_plate.outputs["config"].link(manip_plate.inputConfig)

    manip_car = pipeline.create(dai.node.ImageManip)
    manip_car.inputConfig.setWaitForMessage(True)
    manip_car.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip_car.initialConfig.setResize(72, 72)
    script_car.outputs["passthrough"].link(manip_car.inputImage)
    script_car.outputs["config"].link(manip_car.inputConfig)

    plate_recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    # plate_recognition_nn.setBlobPath(blobconverter.from_zoo(name="license-plate-recognition-barrier-0007"
                                                            # , shaves=shaves, version="2021.4"))
    plate_recognition_nn.setNNArchive(plate_detection_nn_archive)
    manip_plate.out.link(plate_recognition_nn.input)

    car_attribute_nn = pipeline.create(dai.node.NeuralNetwork)
    # car_attribute_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039"
                                                        # , shaves=shaves, version="2021.4"))
    car_attribute_nn.setNNArchive(car_attribute_nn_archive)
    manip_car.out.link(car_attribute_nn.input)

    plate_manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
    plate_detection_nn.out.link(plate_manip_sync.input_detections)
    manip_plate.out.link(plate_manip_sync.input_recognitions)

    car_manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
    car_detection_nn.out.link(car_manip_sync.input_detections)
    manip_car.out.link(car_manip_sync.input_recognitions)

    plate_recognition_sync = pipeline.create(DetectionsRecognitionsSync).build()
    plate_detection_nn.out.link(plate_recognition_sync.input_detections)
    plate_recognition_nn.out.link(plate_recognition_sync.input_recognitions)

    car_attribute_sync = pipeline.create(DetectionsRecognitionsSync).build()
    car_detection_nn.out.link(car_attribute_sync.input_detections)
    car_attribute_nn.out.link(car_attribute_sync.input_recognitions)

    license_plate_recognition = pipeline.create(LicensePlateRecognition).build(
        preview=preview,
        plate_images=plate_manip_sync.output,
        car_images=car_manip_sync.output,
        plate_recognitions=plate_recognition_sync.output,
        car_attributes=car_attribute_sync.output
    )
    license_plate_recognition.inputs["preview"].setBlocking(False)
    license_plate_recognition.inputs["preview"].setMaxSize(16)
    license_plate_recognition.inputs["plate_images"].setBlocking(False)
    license_plate_recognition.inputs["car_images"].setBlocking(False)
    license_plate_recognition.inputs["plate_recognitions"].setBlocking(False)
    license_plate_recognition.inputs["car_attributes"].setBlocking(False)

    print("Pipeline created.")
    pipeline.run()
