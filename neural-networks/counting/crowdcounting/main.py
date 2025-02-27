from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork

from utils.arguments import initialize_argparser
from utils.counter import CrowdCounter
from utils.density_map_transform import DensityMapToFrame
from utils.overlay import OverlayFrames

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device_id)) if args.device_id else dai.Device()
platform = device.getPlatform().name
frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # Model NN Archive
    cc_model_description = dai.NNModelDescription(args.crowd_counting_model)
    cc_model_description.platform = platform
    cc_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(cc_model_description))
    INPUT_WIDTH = cc_model_nn_archive.getInputWidth()
    INPUT_HEIGHT = cc_model_nn_archive.getInputHeight()

    # Video/Camera Input Node
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        replay.setSize(INPUT_WIDTH, INPUT_HEIGHT)
    else:
        cam = pipeline.create(dai.node.Camera).build()
    input_node = replay.out if args.media_path else cam

    # Model Node
    nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, cc_model_nn_archive, fps=args.fps_limit
    )

    # Counter Node
    crowd_counter_node = pipeline.create(CrowdCounter).build(nn.out)

    # Density Map Transform and Resize Nodes
    density_map_transform_node = pipeline.create(DensityMapToFrame).build(nn.out)
    density_map_resize_node = pipeline.create(dai.node.ImageManipV2)
    density_map_resize_node.setMaxOutputFrameSize(INPUT_WIDTH * INPUT_HEIGHT * 3)
    density_map_resize_node.initialConfig.setOutputSize(INPUT_WIDTH, INPUT_HEIGHT)
    density_map_resize_node.initialConfig.setFrameType(frame_type)
    density_map_transform_node.output.link(density_map_resize_node.inputImage)

    # Overlay Frames Node
    overlay_frames = pipeline.create(OverlayFrames).build(
        nn.passthrough, density_map_resize_node.out
    )

    # Visualizer
    visualizer.addTopic("VideoOverlay", overlay_frames.output)
    visualizer.addTopic("Count", crowd_counter_node.output)

    print("Pipeline created.")

    pipeline.start()

    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
