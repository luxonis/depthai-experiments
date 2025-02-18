from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.host_process_detections import ProcessDetections
from utils.host_sync import DetectionsAgeGenderSync
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device) if args.device else dai.DeviceInfo())
platform = device.getPlatform()

FPS = 20
frame_type = dai.ImgFrame.Type.BGR888p
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 30
else:
    raise RuntimeError("This demo is currently only supported on RVC4")

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
        input_node = camera_node.requestOutput((1280, 960), frame_type, fps=FPS)

    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(640, 480)
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)
    input_node.link(resize_node.inputImage)

    face_detection_node: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(resize_node.out, "luxonis/yunet:640x480")

    detection_process_node = pipeline.create(ProcessDetections)
    detection_process_node.set_source_size(1280, 960)
    detection_process_node.set_target_size(62, 62)
    face_detection_node.out.link(detection_process_node.detections_input)

    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(
        Path(__file__).parent / "utils/config_sender_script.py"
    )
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
        crop_node.out, "luxonis/age-gender-recognition:62x62"
    )

    sync_node = pipeline.create(DetectionsAgeGenderSync)
    input_node.link(sync_node.passthrough_input)
    face_detection_node.out.link(sync_node.detections_input)
    age_gender_node.getOutput(0).link(sync_node.age_input)
    age_gender_node.getOutput(1).link(sync_node.gender_input)

    annotation_node = pipeline.create(AnnotationNode)
    sync_node.out.link(annotation_node.input)

    visualizer.addTopic("Video", sync_node.out_frame)
    visualizer.addTopic("Text", annotation_node.output)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
