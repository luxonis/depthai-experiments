import depthai as dai
from utils.host_fatigue_detection import FatigueDetection
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.host_process_detections import ProcessDetections
from pathlib import Path

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
    detection_process_node.set_target_size(192, 192)
    face_detection_node.out.link(detection_process_node.detections_input)

    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(
        Path(__file__).parent / "utils/config_sender_script.py"
    )

    input_node.link(config_sender_node.inputs["frame_input"])
    detection_process_node.config_output.link(config_sender_node.inputs["config_input"])

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(False)

    config_sender_node.outputs["output_config"].link(crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(crop_node.inputImage)

    landmark_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, "luxonis/mediapipe-face-landmarker:192x192"
    )

    fatigue_detection = pipeline.create(FatigueDetection)
    face_detection_node.out.link(fatigue_detection.face_nn)
    face_detection_node.passthrough.link(fatigue_detection.preview)
    landmark_node.out.link(fatigue_detection.landmarks_nn)
    landmark_node.passthrough.link(fatigue_detection.crop_face)

    visualizer.addTopic("Video", face_detection_node.passthrough, "images")
    visualizer.addTopic("Text", fatigue_detection.out, "images")

    print("Pipeline created.")
    pipeline.run()
