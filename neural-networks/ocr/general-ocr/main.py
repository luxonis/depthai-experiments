import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
from utils import OCRAnnotationNode, initialize_argparser, ProcessDetections
from pathlib import Path

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device) if args.device else dai.DeviceInfo())
platform = device.getPlatform()

FPS = 5
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 30
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
        video_resize_node.initialConfig.setFrameType(frame_type)
        replay_node.out.link(video_resize_node.inputImage)

        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1728, 960), frame_type, fps=FPS)

    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(576, 320)
    resize_node.initialConfig.setReusePreviousImage(False)
    input_node.link(resize_node.inputImage)

    detection_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out, "luxonis/paddle-text-detection:320x576"
    )
    detection_node.setNumPoolFrames(30)

    detection_process_node = pipeline.create(ProcessDetections)
    detection_node.out.link(detection_process_node.detections_input)

    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(
        str(Path(__file__).parent / "utils/script_config_sender.py")
    )
    config_sender_node.inputs["frame_input"].setMaxSize(30)
    config_sender_node.inputs["config_input"].setMaxSize(30)
    config_sender_node.inputs["num_configs_input"].setMaxSize(30)

    input_node.link(config_sender_node.inputs["frame_input"])
    detection_process_node.config_output.link(config_sender_node.inputs["config_input"])
    detection_process_node.num_configs_output.link(
        config_sender_node.inputs["num_configs_input"]
    )

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(False)
    crop_node.inputConfig.setMaxSize(30)
    crop_node.inputImage.setMaxSize(30)
    crop_node.setNumFramesPool(30)

    config_sender_node.outputs["output_config"].link(crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(crop_node.inputImage)

    ocr_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, "luxonis/paddle-text-recognition:320x48"
    )
    ocr_node.setNumPoolFrames(30)
    ocr_node.input.setMaxSize(30)

    sync_node = pipeline.create(GatherData).build(FPS)
    detection_process_node.valid_detections.link(sync_node.input_reference)
    ocr_node.out.link(sync_node.input_data)

    annotation_node = pipeline.create(OCRAnnotationNode)
    sync_node.out.link(annotation_node.input)
    detection_node.passthrough.link(annotation_node.passthrough)

    visualizer.addTopic("Video", annotation_node.frame_output)
    visualizer.addTopic("Text", annotation_node.text_annotations_output)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
