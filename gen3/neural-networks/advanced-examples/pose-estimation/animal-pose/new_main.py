import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.process_detections import ProcessDetections
from utils.detection_kpts_sync import DetectionsRecognitionsSync
from utils.annotation_node import AnnotationNode
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
        video_resize_node.initialConfig.setOutputSize(1728, 960)
        video_resize_node.setMaxOutputFrameSize(1728 * 960 * 3)
        video_resize_node.initialConfig.setFrameType(frame_type)
        
        replay_node.out.link(video_resize_node.inputImage)
        
        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1728, 960), frame_type, fps= FPS)   
    
    resize_node = pipeline.create(dai.node.ImageManipV2) 
    resize_node.initialConfig.setOutputSize(512, 288)
    resize_node.initialConfig.setReusePreviousImage(False)
    input_node.link(resize_node.inputImage)
    
    detection_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out, "luxonis/yolov6-nano:r2-coco-512x288"
        )
    
    detection_process_node = pipeline.create(ProcessDetections)
    detection_node.out.link(detection_process_node.detections_input)

    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScript("""
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
    
    input_node.link(config_sender_node.inputs['frame_input'])
    detection_process_node.config_output.link(config_sender_node.inputs['config_input'])
    
    cropNode = pipeline.create(dai.node.ImageManipV2)
    cropNode.inputConfig.setReusePreviousMessage(False)
    cropNode.inputConfig.setMaxSize(20)
    cropNode.inputImage.setReusePreviousMessage(False)
    cropNode.inputImage.setMaxSize(20)
    
    config_sender_node.outputs["output_config"].link(cropNode.inputConfig)
    config_sender_node.outputs['output_frame'].link(cropNode.inputImage)
    
    
    ocr_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        cropNode.out, "luxonis/superanimal-landmarker:256x256"
        )
    ocr_node.input.setBlocking(True)
    ocr_node.input.setMaxSize(20)
    
    sync_node = pipeline.create(DetectionsRecognitionsSync)
    sync_node.recognitions_input.setBlocking(True)
    sync_node.recognitions_input.setMaxSize(20)
    
    ocr_node.out.link(sync_node.recognitions_input)
    detection_node.passthrough.link(sync_node.passthrough_input)
    detection_node.out.link(sync_node.detections_input)
    
    annotation_node = pipeline.create(AnnotationNode)
    sync_node.out.link(annotation_node.input)
    
    visualizer.addTopic("Video", detection_node.passthrough, "images")
    visualizer.addTopic("Detections", annotation_node.out_detections, "images")
    # visualizer.addTopic("Text", annotation_node.out_keypoints, "images")
    
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord('q'):
            print("Got q key. Exiting...")
            break