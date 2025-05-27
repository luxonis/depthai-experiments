from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from util.arguments import initialize_argparser
from util.crop_face import CropFace

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription("luxonis/yunet:320x240")
    platform = device.getPlatformAsString()
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
        cam_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            (1920, 1080), dai.ImgFrame.Type.NV12, fps=args.fps_limit
        )

    image_manip = pipeline.create(dai.node.ImageManip)
    image_manip.setMaxOutputFrameSize(
        nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
    )
    image_manip.initialConfig.setOutputSize(
        nn_archive.getInputWidth(), nn_archive.getInputHeight()
    )
    image_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    if platform == "RVC4":
        image_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
    cam_out.link(image_manip.inputImage)

    nn_with_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        image_manip.out, nn_archive
    )

    crop_face = pipeline.create(CropFace).build(nn_with_parser.out)
    crop_manip = pipeline.create(dai.node.ImageManip)
    crop_manip.inputConfig.setWaitForMessage(False)
    crop_manip.setMaxOutputFrameSize(1920 * 1080 * 3)
    crop_face.output.link(crop_manip.inputConfig)
    cam_out.link(crop_manip.inputImage)

    encoder = pipeline.create(dai.node.VideoEncoder)  # only works on RVC4
    encoder.setDefaultProfilePreset(30.0, dai.VideoEncoderProperties.Profile.H264_MAIN)
    cam_out.link(encoder.input)

    visualizer.addTopic("Video", encoder.out, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    visualizer.addTopic("Cropped Face", crop_manip.out, "crop")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
