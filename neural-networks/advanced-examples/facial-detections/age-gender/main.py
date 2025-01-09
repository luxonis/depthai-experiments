from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.host_process_detections import ProcessDetections
from utils.host_sync import DetectionsRecognitionsSync

_, args = initialize_argparser()
# visualizer = dai.RemoteConnection(httpPort=8082)
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
        video_resize_node.initialConfig.setOutputSize(1280, 960)
        video_resize_node.initialConfig.setFrameType(frame_type)
        
        replay_node.out.link(video_resize_node.inputImage)
        
        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1280, 960), frame_type, fps= FPS)   
    
    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(640, 480)
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)
    input_node.link(resize_node.inputImage)
    
    face_detection_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out, "luxonis/yunet:new-480x640"
    )

    detection_process_node = pipeline.create(ProcessDetections)
    detection_process_node.set_source_size(1280, 960)
    detection_process_node.set_target_size(62, 62)
    face_detection_node.out.link(detection_process_node.detections_input)
    
    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(Path(__file__).parent / "utils/config_sender_script.py")
    config_sender_node.inputs["frame_input"].setBlocking(True)
    config_sender_node.inputs["config_input"].setBlocking(True)
    config_sender_node.inputs["frame_input"].setMaxSize(30)
    config_sender_node.inputs["config_input"].setMaxSize(30)
    
    input_node.link(config_sender_node.inputs["frame_input"])
    detection_process_node.config_output.link(config_sender_node.inputs["config_input"])
    
    
    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(False)
    
    config_sender_node.outputs["output_config"].link(crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(crop_node.inputImage)
    

    age_gender_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, "luxonis/age-gender-recognition:new-62x62"
        )

    
    sync = pipeline.create(DetectionsRecognitionsSync)
    input_node.link(sync.passthrough_input)
    face_detection_node.out.link(sync.detections_input)
    age_gender_node.out.link(sync.recognitions_input)
    
    output_q = sync.out.createOutputQueue()
    
    print("Pipeline created.")
    pipeline.start()

    while pipeline.isRunning():
        msg = output_q.get()
        frame = msg["passthrough"].getCvFrame()
        det_msg = msg["detections"]
        rec_msg = msg["recognitions"]
        
        frame_ts = frame.getTimestamp()
        det_ts = det_msg.getTimestamp()
        rec_ts = rec_msg.getTimestamp()
        print(f"Frame ts: {frame_ts}, Det ts: {det_ts}, Rec ts: {rec_ts}")
        
        for detection, recognition in zip(det_msg.detections, rec_msg.recognitions):
            print(detection)
            print("rec", recognition)
            
        