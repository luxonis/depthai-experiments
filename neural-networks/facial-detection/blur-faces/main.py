import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from pathlib import Path
from utils.blur_detections import BlurBboxes

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

MODEL = "luxonis/yunet:640x480"

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # face detection model
    det_model_description = dai.NNModelDescription(MODEL)
    det_model_description.platform = platform
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # media/camera input
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(
            dai.ImgFrame.Type.BGR888i
            if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )
        replay.setLoop(True)
        replay.setSize(
            det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
        )
    input_node = replay if args.media_path else pipeline.create(dai.node.Camera).build()

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_model_nn_archive, fps=args.fps_limit
    )

    # blurring
    blur_node = pipeline.create(BlurBboxes)
    det_nn.out.link(blur_node.input_detections)
    det_nn.passthrough.link(blur_node.input_frame)

    # visualization
    visualizer.addTopic("Video", det_nn.passthrough)
    visualizer.addTopic("FaceBlur", blur_node.out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
