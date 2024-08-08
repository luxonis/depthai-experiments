import argparse as argparse
import depthai as dai
import time
import math
import asyncio
import cv2
import numpy as np
from pathlib import Path

from foxglove_utils import create_channels, send_frame, send_pointcloud, Listener
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer

from projector_device import PointCloudVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--left', default=False, action="store_true", help="enable left camera stream")
parser.add_argument('-r', '--right', default=False, action="store_true", help="enable right camera stream")
parser.add_argument('-pc', '--pointcloud', default=False, action="store_true", help="enable pointcloud stream")
parser.add_argument('-nc', '--no-color', default=False, action="store_true", help="disable color camera stream")
args = parser.parse_args()

resolution = (640, 400)
# To speed up visualization downsample the pointcloud before operating on it and visualizing
downsample_pcl = True
process_pointcloud = PointCloudVisualizer() if args.pointcloud else None

async def main():
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

        seq = 0
        if args.pointcloud:
            calibration = device.readCalibration()
            M_right = calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, dai.Size2f(resolution[0], resolution[1]))
            xyz = create_xyz(resolution[0], resolution[1], np.array(M_right))

            buffer = dai.NNData()
            buffer.addTensor("0", xyz)

            xyz_queue = nn.inputs["xyz"].createInputQueue()
            xyz_queue.send(buffer)

        # Host node architecture is not good for this experiment because we want to call async functions to
        #  communicate with the foxglove server
        color_q = cam.isp.createOutputQueue(blocking=False, maxSize=1)
        left_q = left.out.createOutputQueue(blocking=False, maxSize=1)
        right_q = right.out.createOutputQueue(blocking=False, maxSize=1)
        pcl_q = nn.out.createOutputQueue(blocking=False, maxSize=8)

        print("Pipeline created.")
        pipeline.start()

        # Start server and wait for foxglove connection
        async with FoxgloveServer("0.0.0.0", 8765, "DepthAI server") as server:
            server.set_listener(Listener())

            color_channel, pointcloud_channel, left_channel, right_channel \
                = await create_channels(server, not args.no_color, args.pointcloud, args.left, args.right)

            while pipeline.isRunning():
                tmpTime = time.time_ns()
                sec = math.trunc(tmpTime / 1e9)
                nsec = tmpTime - sec

                await asyncio.sleep(0.1)

                if not args.no_color and color_q.has():
                    frame = color_q.get().getCvFrame()
                    await send_frame(server, frame, sec, nsec, color_channel)

                if args.left and left_q.has():
                    frame = left_q.get().getCvFrame()
                    await send_frame(server, frame, sec, nsec, left_channel)

                if args.right and right_q.has():
                    frame = right_q.get().getCvFrame()
                    await send_frame(server, frame, sec, nsec, right_channel)

                if args.pointcloud and pcl_q.has():
                    pointcloud = pcl_q.get()

                    pcl_data = pointcloud.getFirstTensor().flatten().reshape(1, 3, resolution[0], resolution[1])
                    pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0

                    process_pointcloud.visualize_pcl(pcl_data, downsample=downsample_pcl)
                    pcl_data = process_pointcloud.pcl.points

                    await send_pointcloud(server, pcl_data, sec, nsec, pointcloud_channel, seq)
                    seq += 1

                if cv2.waitKey(1) == ord('q'):
                    print("Pipeline exited.")
                    break


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

if __name__ == "__main__":
    run_cancellable(main())
