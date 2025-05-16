import depthai as dai

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
from depthai_nodes.node import ApplyColormap

_, args = initialize_argparser()

model_reference: str = args.model

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

platform = device.getPlatform().name
print(f"Platform: {platform}")

if args.fps_limit is None:
    args.fps_limit = 20 if platform == "RVC2" else 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

model_description = dai.NNModelDescription(model_reference)
model_description.platform = platform
nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

with dai.Pipeline(device) as pipeline:
    # Check if the device has color, left and right cameras
    available_cameras = device.getConnectedCameras()

    if len(available_cameras) < 3:
        raise ValueError(
            "Device must have 3 cameras (color, left and right) in order to run this experiment."
        )

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestOutput(nn_archive.getInputSize(), fps=args.fps_limit),
        right=right_cam.requestOutput(nn_archive.getInputSize(), fps=args.fps_limit),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    if platform == "RVC2":
        stereo.setOutputSize(*nn_archive.getInputSize())
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)

    nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=cam,
        stereo=stereo,
        nnArchive=nn_archive,
        fps=float(args.fps_limit),
    )
    if platform == "RVC2":
        nn.setNNArchive(
            nn_archive, numShaves=7
        )  # TODO: change to numShaves=4 if running on OAK-D Lite
    nn.setBoundingBoxScaleFactor(0.7)

    annotation_node = pipeline.create(AnnotationNode).build(
        input_detections=nn.out,
        depth=stereo.depth,
        labels=nn_archive.getConfig().model.heads[0].metadata.classes,
    )

    apply_colormap = pipeline.create(ApplyColormap).build(stereo.depth)

    visualizer.addTopic("Camera", nn.passthrough)
    visualizer.addTopic("Detections", annotation_node.out_annotations)
    visualizer.addTopic("Depth", apply_colormap.out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            pipeline.stop()
            break
    print("Pipeline finished.")
