from pathlib import Path
import depthai as dai
import numpy as np

from depthai_nodes.node import ParsingNeuralNetwork
from utils.host_pointcloud_display import PointcloudDisplay
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

DEPTH_SHAPE = (640, 400)


def create_xyz(width, height, camera_matrix):
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # Generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
    points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

    # Unpack coordinates
    u_coord: np.array = points_2d[..., 0]
    v_coord: np.array = points_2d[..., 1]

    # Unpack intrinsics
    fx: np.array = camera_matrix[0, 0]
    fy: np.array = camera_matrix[1, 1]
    cx: np.array = camera_matrix[0, 2]
    cy: np.array = camera_matrix[1, 2]

    # Projective
    x_coord: np.array = (u_coord - cx) / fx
    y_coord: np.array = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord], axis=-1)
    return np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)


visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    platform = device.getPlatform()

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        DEPTH_SHAPE,
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = left.requestOutput(DEPTH_SHAPE)

    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = right.requestOutput(DEPTH_SHAPE)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_out,
        right=right_out,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(DEPTH_SHAPE[0], DEPTH_SHAPE[1])

    """ In-place post-processing configuration for a stereo depth node
    The best combo of filters is application specific. Hard to say there is a one size fits all.
    They also are not free. Even though they happen on device, you pay a penalty in fps. """
    stereo.initialConfig.postProcessing.speckleFilter.enable = True
    stereo.initialConfig.postProcessing.speckleFilter.speckleRange = 60
    stereo.initialConfig.postProcessing.temporalFilter.enable = True
    stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 2
    stereo.initialConfig.postProcessing.spatialFilter.numIterations = 1
    stereo.initialConfig.postProcessing.thresholdFilter.minRange = 700  # mm
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 4000  # mm
    stereo.initialConfig.censusTransform.enableMeanMode = True
    stereo.initialConfig.costMatching.linearEquationParameters.alpha = 0
    stereo.initialConfig.costMatching.linearEquationParameters.beta = 2

    calibration = device.readCalibration()
    nn_blob_path = Path(__file__).parent / "model/pointcloud_640x400.blob"
    nn_blob_path = nn_blob_path.resolve().absolute()
    if not nn_blob_path.exists():
        raise FileNotFoundError(f"Blob file not found at {nn_blob_path}")

    try:
        nn_archive = dai.NNArchive(str(nn_blob_path))

        parser = pipeline.create(ParsingNeuralNetwork).build(
            input=stereo.depth, nn_source=nn_archive
        )
    except Exception as e:
        print(f"Failed to load blob file: {e}")
        print("Possible solutions:")
        print("1. Verify the blob file is not corrupted")
        print("2. Recompile the model with the latest depthai version")
        print("3. Check the blob was compiled for the correct MyriadX version")
        raise

    # Only send xyz data once, and always reuse the message
    parser.inputs["xyz"].setReusePreviousMessage(True)
    # Get xyz data and send it to the device - to the pointcloud neural network
    M_right = calibration.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_C, dai.Size2f(DEPTH_SHAPE[0], DEPTH_SHAPE[1])
    )
    xyz = create_xyz(DEPTH_SHAPE[0], DEPTH_SHAPE[1], np.array(M_right))
    buffer = dai.NNData()
    buffer.addTensor("0", xyz)
    xyz_queue = parser.inputs["xyz"].createInputQueue()
    xyz_queue.send(buffer)

    display = pipeline.create(PointcloudDisplay).build(
        preview=cam_out,
        pointcloud=parser.out,
        depth_shape=DEPTH_SHAPE,
    )
    display.inputs["preview"].setBlocking(False)
    display.inputs["pointcloud"].setBlocking(False)
    display.inputs["pointcloud"].setMaxSize(8)

    visualizer.addTopic("Main Stream", display.passthrough, "images")
    # visualizer.addTopic("Pointcloud", display.annotation_output, "images")

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
