import rerun as rr
import numpy as np
import depthai as dai
import cv2
import asyncio
import subprocess

subprocess.Popen(["rerun", "--memory-limit", "200MB"])

pipeline = dai.Pipeline()
lrcheck = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
COLOR = True

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(resolution)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(resolution)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)

stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
stereo.initialConfig.setSubpixelFractionalBits(5)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 200000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

xout_colorize = pipeline.createXLinkOut()
xout_colorize.setStreamName("colorize")

if COLOR:
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    camRgb.isp.link(xout_colorize.input)
else:
    stereo.rectifiedRight.link(xout_colorize.input)


def create_xyz(device, width, height):
    calibData = device.readCalibration()
    M_right = calibData.getCameraIntrinsics(
        dai.CameraBoardSocket.RIGHT, dai.Size2f(width, height))
    camera_matrix = np.array(M_right).reshape(3, 3)

    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
    points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

    # unpack coordinates
    u_coord: np.array = points_2d[..., 0]
    v_coord: np.array = points_2d[..., 1]

    # unpack intrinsics
    fx: np.array = camera_matrix[0, 0]
    fy: np.array = camera_matrix[1, 1]
    cx: np.array = camera_matrix[0, 2]
    cy: np.array = camera_matrix[1, 2]

    # projective
    x_coord: np.array = (u_coord - cx) / fx
    y_coord: np.array = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord], axis=-1)
    return np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)


async def main():
    with dai.Device(pipeline) as device:
        print("Opening device", resolution)
        q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
        q_colorize = device.getOutputQueue(
            "colorize", maxSize=1, blocking=False)
        xyz = None  # Just so it's more clear
        while True:
            if q_depth.has() and q_colorize.has():
                depth_frame = q_depth.get().getCvFrame()
                if xyz is None:
                    xyz = create_xyz(device, depth_frame.shape[1],
                                     depth_frame.shape[0])
                depth_frame = np.expand_dims(np.array(depth_frame), axis=-1)
                # To meters and reshape for rerun
                pcl = (xyz * depth_frame / 1000.0).reshape(-1, 3)
                colors = cv2.cvtColor(
                    q_colorize.get().getCvFrame(), cv2.COLOR_BGR2RGB).reshape(-1, 3)
                rr.log_points("Pointcloud", pcl,
                              colors=colors)

if __name__ == "__main__":
    rr.init("Rerun ", spawn=False)
    rr.connect()
    asyncio.run(main())
