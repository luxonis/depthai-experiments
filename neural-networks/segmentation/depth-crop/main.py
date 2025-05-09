import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.annotation_node import AnnotationNode

from utils.arguments import initialize_argparser

_, args = initialize_argparser()

model_reference_rvc2 = "luxonis/deeplab-v3-plus:256x256"
model_reference_rvc4 = "luxonis/deeplab-v3-plus:512x288"


visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

CAMERA_RESOLUTION = (640, 400)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    platform = device.getPlatform().name
    FPS_LIMIT = 10 if platform == "RVC2" else 20
    print(f"Platform: {platform}")

    # Check if the device has color, left and right cameras
    available_cameras = device.getConnectedCameras()

    if len(available_cameras) < 3:
        raise ValueError(
            "Device must have 3 cameras (color, left and right) in order to run this experiment."
        )

    model_description = dai.NNModelDescription(
        model=model_reference_rvc2 if platform == "RVC2" else model_reference_rvc4,
    )
    model_description.platform = platform
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_output = color.requestOutput(
        CAMERA_RESOLUTION, dai.ImgFrame.Type.NV12, fps=FPS_LIMIT
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(CAMERA_RESOLUTION, fps=FPS_LIMIT),
        right=right.requestOutput(CAMERA_RESOLUTION, fps=FPS_LIMIT),
    )
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    if platform == "RVC2":
        stereo.setOutputSize(*CAMERA_RESOLUTION)

    manip = pipeline.create(dai.node.ImageManipV2)
    manip.initialConfig.setOutputSize(*nn_archive.getInputSize())
    manip.initialConfig.setFrameType(
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )
    manip.setMaxOutputFrameSize(
        nn_archive.getInputSize()[0] * nn_archive.getInputSize()[1] * 3
    )
    color_output.link(manip.inputImage)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        nn_source=nn_archive, input=manip.out
    )

    annotation_node = pipeline.create(AnnotationNode).build(
        preview=color_output,
        disparity=stereo.disparity,
        mask=nn.out,
        max_disparity=stereo.initialConfig.getMaxDisparity(),
    )

    visualizer.addTopic("Segmentation", annotation_node.output_segmentation)
    visualizer.addTopic("Cutout", annotation_node.output_cutout)
    visualizer.addTopic("Depth", annotation_node.output_depth)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
