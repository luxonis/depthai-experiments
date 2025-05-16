import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from pathlib import Path
from utils.blur_detections import BlurBboxes


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

if args.fps_limit is None:
    args.fps_limit = 10 if platform == "RVC2" else 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    if args.media_path:
        replay_node = pipeline.create(dai.node.ReplayVideo)
        replay_node.setReplayVideoFile(Path(args.media_path))
        replay_node.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay_node.setLoop(True)
        replay_node.setFps(args.fps_limit)

        video_resize_node = pipeline.create(dai.node.ImageManipV2)
        video_resize_node.initialConfig.setOutputSize(576, 320)
        video_resize_node.initialConfig.setFrameType(frame_type)

        replay_node.out.link(video_resize_node.inputImage)

        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput(
            (576, 320), frame_type, fps=args.fps_limit
        )

    detection_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, "luxonis/paddle-text-detection:320x576"
    )

    blur_node = pipeline.create(BlurBboxes)
    detection_node.out.link(blur_node.input_detections)
    detection_node.passthrough.link(blur_node.input_frame)

    visualizer.addTopic("Video", detection_node.passthrough)
    visualizer.addTopic("Blur Text", blur_node.out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
