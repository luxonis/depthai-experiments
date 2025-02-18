import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from pathlib import Path
from utils.blur_detections import BlurBboxes


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device) if args.device else dai.DeviceInfo())
platform = device.getPlatform()

FPS = 28
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 28
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
        video_resize_node.initialConfig.setOutputSize(576, 320)
        video_resize_node.initialConfig.setFrameType(frame_type)

        replay_node.out.link(video_resize_node.inputImage)

        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((576, 320), frame_type, fps=FPS)

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
