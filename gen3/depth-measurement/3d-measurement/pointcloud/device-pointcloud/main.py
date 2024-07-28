import depthai as dai
from pathlib import Path
import numpy as np

from host_pointcloud_display import PointcloudDisplay

depth_shape = (640, 400)

def create_xyz(width, height, camera_matrix):
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # Generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys)) # WxHx2
    points_2d = base_grid.transpose(1, 2, 0) # 1xHxWx2

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
    return np.pad(xyz, ((0,0),(0,0),(0,1)), "constant", constant_values=1.0)

device = dai.Device()
with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setIspScale(1, 3)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

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

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(Path("model/pointcloud_640x400.blob").resolve().absolute())
    stereo.depth.link(nn.inputs["depth"])
    # Only send xyz data once, and always reuse the message
    nn.inputs["xyz"].setReusePreviousMessage(True)

    # Get xyz data and send it to the device - to the pointcloud neural network
    calibration = device.readCalibration()
    M_right = calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, dai.Size2f(depth_shape[0], depth_shape[1]))
    xyz = create_xyz(depth_shape[0], depth_shape[1], np.array(M_right))
    buffer = dai.NNData()
    buffer.addTensor("0", xyz)
    xyz_queue = nn.inputs["xyz"].createInputQueue()
    xyz_queue.send(buffer)

    display = pipeline.create(PointcloudDisplay).build(
        preview=cam.isp,
        pointcloud=nn.out,
        depth_shape=depth_shape
    )
    display.inputs["preview"].setBlocking(False)
    display.inputs["pointcloud"].setBlocking(False)
    display.inputs["pointcloud"].setMaxSize(8)

    print("Pipeline created.")
    pipeline.run()
