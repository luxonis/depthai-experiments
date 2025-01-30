import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.depth_merger import DepthMerger
from utils.host_bird_eye_view import BirdsEyeView
from utils.measure_object_distance import MeasureObjectDistance
from host_social_distancing import SocialDistancing
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
visualizer = dai.RemoteConnection(httpPort=8082)

platform = device.getPlatform().name
print(f"Platform: {platform}")

FPS = 10 if platform == "RVC2" else 20

modelDescription = dai.NNModelDescription("luxonis/scrfd-person-detection:25g-640x640")
modelDescription.platform = platform
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)
nnArchive = dai.NNArchive(archivePath)


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    rgb = cam.requestOutput(
        size=nnArchive.getInputSize(),
        type=dai.ImgFrame.Type.BGR888p
        if platform == "RVC2"
        else dai.ImgFrame.Type.BGR888i,
        fps=FPS,
    )

    left = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_B
    )
    right = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_C
    )

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(nnArchive.getInputSize(), fps=FPS),
        right=right.requestOutput(nnArchive.getInputSize(), fps=FPS),
    )
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    if platform == "RVC2":
        stereo.setOutputSize(*nnArchive.getInputSize())

    nn_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input=rgb,
        nn_source=nnArchive,
    )

    depth_merger = pipeline.create(DepthMerger).build(
        output_2d=nn_parser.out,
        output_depth=stereo.depth,
        calib_data=device.readCalibration2(),
        depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
    )

    bird_eye_view = pipeline.create(BirdsEyeView).build(depth_merger.output)

    measure_obj_dist = pipeline.create(MeasureObjectDistance).build(depth_merger.output)

    social_distancing = pipeline.create(SocialDistancing).build(
        distances=measure_obj_dist.output
    )

    visualizer.addTopic("Video", rgb, "images")
    visualizer.addTopic("Detections", nn_parser.out, "images")
    visualizer.addTopic("Distances", social_distancing.output, "images")
    visualizer.addTopic("Bird-eye view", bird_eye_view.output, "bird-eye")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
print("Pipeline stopped.")
