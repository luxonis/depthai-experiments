import time
from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.filter_classes import FilterClasses
from utils.visualizer import CustomVisualizer
from datetime import timedelta

_, args = initialize_argparser()

detection_model_slug = "luxonis/yolov6-nano:r2-coco-512x288"
pose_model_slug = "luxonis/superanimal-landmarker:256x256"

if args.fps_limit and args.media_path:
    args.fps_limit = None
    print(
        "WARNING: FPS limit is set but media path is provided. FPS limit will be ignored."
    )

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(detection_model_slug)
    platform = device.getPlatform().name
    print(f"Platform: {platform}")
    detection_model_description.platform = platform
    detection_nn_archive = dai.NNArchive(dai.getModelFromZoo(detection_model_description))
    classes = detection_nn_archive.getConfig().model.heads[0].metadata.classes

    pose_model_description = dai.NNModelDescription(pose_model_slug)
    pose_model_description.platform = platform
    pose_nn_archive = dai.NNArchive(dai.getModelFromZoo(pose_model_description, useCached=False))
    connection_pairs = pose_nn_archive.getConfig().model.heads[0].metadata.extraParams["connection_pairs"]

    frame_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        replay.setFps(4)
        imageManip = pipeline.create(dai.node.ImageManipV2)
        imageManip.setMaxOutputFrameSize(
            detection_nn_archive.getInputWidth() * detection_nn_archive.getInputHeight() * 3
        )
        imageManip.initialConfig.addResize(
            detection_nn_archive.getInputWidth(), detection_nn_archive.getInputHeight()
        )
        imageManip.initialConfig.setFrameType(frame_type)
        if platform == "RVC4":
            imageManip.initialConfig.setFrameType(frame_type)
        replay.out.link(imageManip.inputImage)
    
    else:
        cam = pipeline.create(dai.node.Camera).build()#.requestOutput(detection_nn_archive.getInputSize(), type=frame_type, fps=args.fps_limit)
    input_node = (
        imageManip.out if args.media_path else cam
    )

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, detection_nn_archive, fps=args.fps_limit
    )

    filter_classes = pipeline.create(FilterClasses, labels=[0, 15, 16, 17, 18, 19, 20, 21, 22, 23])

    detection_nn.out.link(filter_classes.input_detections)

    script = pipeline.create(dai.node.Script)
    filter_classes.out.link(script.inputs['det_in'])
    detection_nn.passthrough.link(script.inputs['preview'])
    script_filename = "script_rvc2.py" if platform == "RVC2" else "script_rvc4.py"
    script.setScriptPath(Path(__file__).parent / script_filename)

    if platform == "RVC2":
        pose_manip = pipeline.create(dai.node.ImageManip)
        pose_manip.initialConfig.setResize(256, 256)
        pose_manip.inputConfig.setWaitForMessage(True)
    elif platform == "RVC4":
        pose_manip = pipeline.create(dai.node.ImageManipV2)
        pose_manip.initialConfig.addResize(256, 256)
        pose_manip.inputConfig.setWaitForMessage(True)
    script.outputs['manip_cfg'].link(pose_manip.inputConfig)
    script.outputs['manip_img'].link(pose_manip.inputImage)

    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(pose_manip.out, pose_nn_archive)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=50))
    sync.setRunOnHost(True)
    detection_nn.out.link(sync.inputs["detections"])
    pose_nn.out.link(sync.inputs["keypoints"])

    custom_visualizer = pipeline.create(CustomVisualizer, connection_pairs=connection_pairs)
    sync.out.link(custom_visualizer.input)

    visualizer.addTopic("Video", detection_nn.passthrough, "images")
    visualizer.addTopic("Detections", custom_visualizer.out_detections, "images")
    visualizer.addTopic("Pose", custom_visualizer.out_keypoints, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        time.sleep(1 / 30)
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break