import depthai as dai
from depthai_nodes.node import ApplyColormap

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

if args.fps_limit is None:
    args.fps_limit = 20 if platform == "RVC2" else 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )


frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # Check if the device has color, left and right cameras
    available_cameras = device.getConnectedCameras()
    if len(available_cameras) < 3:
        raise ValueError(
            "Device must have 3 cameras (color, left and right) in order to run this experiment."
        )

    # detection model
    det_model_description = dai.NNModelDescription(args.model)
    det_model_description.platform = platform
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # camera input
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestOutput(
            det_model_nn_archive.getInputSize(), fps=args.fps_limit
        ),
        right=right_cam.requestOutput(
            det_model_nn_archive.getInputSize(), fps=args.fps_limit
        ),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    if platform == "RVC2":
        stereo.setOutputSize(*det_model_nn_archive.getInputSize())
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)

    nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=cam,
        stereo=stereo,
        nnArchive=det_model_nn_archive,
        fps=float(args.fps_limit),
    )
    if platform == "RVC2":
        nn.setNNArchive(
            det_model_nn_archive, numShaves=7
        )  # TODO: change to numShaves=4 if running on OAK-D Lite
    nn.setBoundingBoxScaleFactor(0.7)

    # annotation
    annotation_node = pipeline.create(AnnotationNode).build(
        input_detections=nn.out,
        depth=stereo.depth,
        labels=det_model_nn_archive.getConfig().model.heads[0].metadata.classes,
    )

    apply_colormap = pipeline.create(ApplyColormap).build(stereo.depth)

    # visualization
    visualizer.addTopic("Camera", nn.passthrough)
    visualizer.addTopic("Detections", annotation_node.out_annotations)
    visualizer.addTopic("Depth", apply_colormap.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
