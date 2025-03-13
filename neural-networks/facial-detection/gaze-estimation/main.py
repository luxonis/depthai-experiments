from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.process_keypoints import LandmarksProcessing
from utils.node_creators import create_crop_node, create_gaze_estimation_model
from utils.host_sync import ImageLandmarkSync
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform()

FPS = 30
frame_type = dai.ImgFrame.Type.BGR888i
if "RVC2" in str(platform):
    raise RuntimeError(
        f"This demo is currently only supported on RVC4, got `{platform}`"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    if args.media_path:
        replay_node = pipeline.create(dai.node.ReplayVideo)
        replay_node.setReplayVideoFile(Path(args.media_path))
        replay_node.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay_node.setLoop(True)

        video_resize_node = pipeline.create(dai.node.ImageManipV2)
        video_resize_node.initialConfig.setOutputSize(1280, 1280)
        video_resize_node.initialConfig.setFrameType(frame_type)

        replay_node.out.link(video_resize_node.inputImage)

        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1280, 1280), frame_type, fps=FPS)

    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(640, 640)
    resize_node.setMaxOutputFrameSize(640 * 640 * 3)
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)
    input_node.link(resize_node.inputImage)

    face_detection_node: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(resize_node.out, "luxonis/scrfd-face-detection:10g-640x640")
    face_detection_node.input.setBlocking(True)
    detection_process_node = pipeline.create(LandmarksProcessing)
    detection_process_node.set_source_size(1280, 1280)
    detection_process_node.set_target_size(60, 60)
    face_detection_node.out.link(detection_process_node.detections_input)

    left_eye_crop_node = create_crop_node(
        pipeline, input_node, detection_process_node.left_config_output
    )
    right_eye_crop_node = create_crop_node(
        pipeline, input_node, detection_process_node.right_config_output
    )
    face_crop_node = create_crop_node(
        pipeline, input_node, detection_process_node.face_config_output
    )

    head_pose_node = pipeline.create(dai.node.NeuralNetwork)
    head_pose_node.setFromModelZoo(
        dai.NNModelDescription("luxonis/head-pose-estimation:60x60"), useCached=True
    )
    head_pose_node.input.setBlocking(True)
    face_crop_node.out.link(head_pose_node.input)

    head_pose_script = pipeline.create(dai.node.Script)
    head_pose_script.setScriptPath(Path(__file__).parent / "utils/head_pose_script.py")
    head_pose_node.out.link(head_pose_script.inputs["pose_input"])
    head_pose_script.inputs["pose_input"].setBlocking(True)

    gaze_estimation_node = create_gaze_estimation_model(
        pipeline,
        head_pose_script.outputs["head_pose_output"],
        left_eye_crop_node.out,
        right_eye_crop_node.out,
    )

    host_sync_node = pipeline.create(ImageLandmarkSync)
    gaze_estimation_node.out.link(host_sync_node.gaze_input)
    face_detection_node.out.link(host_sync_node.detections_input)
    face_detection_node.passthrough.link(host_sync_node.frame_input)
    host_sync_node.frame_input.setBlocking(True)
    host_sync_node.gaze_input.setBlocking(True)
    host_sync_node.detections_input.setBlocking(True)
    host_sync_node.frame_input.setMaxSize(5)
    host_sync_node.gaze_input.setMaxSize(5)
    host_sync_node.detections_input.setMaxSize(5)

    annotation_node = pipeline.create(AnnotationNode)
    host_sync_node.out.link(annotation_node.input)

    visualizer.addTopic("annotation", annotation_node.output)
    visualizer.addTopic("frame", host_sync_node.frame_out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
