from pathlib import Path

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.mjpeg_streamer import MJPEGStreamer

_, args = initialize_argparser()


device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription("luxonis/yolov6-nano:r2-coco-512x288")
    platform = pipeline.getDefaultDevice().getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        replay.setSize(1920, 1080)
        image_manip = pipeline.create(dai.node.ImageManipV2)
        image_manip.setMaxOutputFrameSize(
            nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
        )
        image_manip.initialConfig.setOutputSize(
            nn_archive.getInputWidth(), nn_archive.getInputHeight()
        )
        image_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if platform == "RVC4":
            image_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        replay.out.link(image_manip.inputImage)

    input_node = (
        image_manip.out if args.media_path else pipeline.create(dai.node.Camera).build()
    )

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    mjpeg_streamer = pipeline.create(MJPEGStreamer).build(
        preview=nn_with_parser.passthrough,
        nn=nn_with_parser.out,
        labels=nn_archive.getConfigV1().model.heads[0].metadata.classes,
    )

    pipeline.run()
