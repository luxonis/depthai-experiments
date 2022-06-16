#!/usr/bin/env python3
import asyncio
import base64
import json
import struct
import time
import math

import argparse as argparse
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import os
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId

try:
    from projector_device import PointCloudVisualizer
except ImportError as e:
    raise ImportError(
        f"\033[1;5;31mError occured when importing PCL projector: {e}")

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--left', default=False, action="store_true", help="Enable streaming from left camera")
parser.add_argument('-r', '--right', default=False, action="store_true", help="Enable streaming from right camera")
parser.add_argument('-dpcl', '--disable_pcl', default=False, action="store_true", help="Disable streaming point cloud from camera")
parser.add_argument('-dc', '--disable_color', default=False, action="store_true", help="Disable streaming color camera")
args = parser.parse_args()
print(args)
# Depth resolution
resolution = (640, 400)  # 24 FPS (without visualization)

# parameters to speed up visualization
downsample_pcl = True  # downsample the pointcloud before operating on it and visualizing

# StereoDepth config options.
# whether or not to align the depth image on host (As opposed to on device), only matters if align_depth = True
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels


def create_xyz(width, height, camera_matrix):
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


def getPath(resolution):
    (width, heigth) = resolution
    path = Path("models", "out")
    path.mkdir(parents=True, exist_ok=True)
    name = f"pointcloud_{width}x{heigth}"

    return_path = str(path / (name + '.blob'))
    if os.path.exists(return_path):
        return return_path


def configureDepthPostProcessing(stereoDepthNode):
    """
    In-place post-processing configuration for a stereo depth node
    The best combo of filters is application specific. Hard to say there is a one size fits all.
    They also are not free. Even though they happen on device, you pay a penalty in fps.
    """
    stereoDepthNode.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # stereoDepthNode.initialConfig.setBilateralFilterSigma(16)
    config = stereoDepthNode.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 60
    config.postProcessing.temporalFilter.enable = True

    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 700  # mm
    config.postProcessing.thresholdFilter.maxRange = 4000  # mm
    # config.postProcessing.decimationFilter.decimationFactor = 1
    config.censusTransform.enableMeanMode = True
    config.costMatching.linearEquationParameters.alpha = 0
    config.costMatching.linearEquationParameters.beta = 2
    stereoDepthNode.initialConfig.set(config)
    stereoDepthNode.setLeftRightCheck(lrcheck)
    stereoDepthNode.setExtendedDisparity(extended)
    stereoDepthNode.setSubpixel(subpixel)
    stereoDepthNode.setRectifyEdgeFillColor(0)  # Black, to better see the cutout


def get_resolution(width):
    if width == 480:
        return dai.MonoCameraProperties.SensorResolution.THE_480_P
    elif width == 720:
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    elif width == 800:
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    else:
        return dai.MonoCameraProperties.SensorResolution.THE_400_P

pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
camRgb.setIspScale(1, 3)

rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("rgb")
camRgb.isp.link(rgbOut.input)

# Configure Camera Properties
left = pipeline.createMonoCamera()
left.setResolution(get_resolution(resolution[1]))
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

# left camera output
leftOut = pipeline.createXLinkOut()
leftOut.setStreamName("left")
left.out.link(leftOut.input)

right = pipeline.createMonoCamera()
right.setResolution(get_resolution(resolution[1]))
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# right camera output
rightOut = pipeline.createXLinkOut()
rightOut.setStreamName("right")
right.out.link(rightOut.input)

stereo = pipeline.createStereoDepth()
# configureDepthPostProcessing(stereo)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Depth -> PointCloud
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(getPath(resolution))
stereo.depth.link(nn.inputs["depth"])

xyz_in = pipeline.createXLinkIn()
xyz_in.setMaxDataSize(6144000)
xyz_in.setStreamName("xyz_in")
xyz_in.out.link(nn.inputs["xyz"])

# Only send xyz data once, and always reuse the message
nn.inputs["xyz"].setReusePreviousMessage(True)

pointsOut = pipeline.createXLinkOut()
pointsOut.setStreamName("pcl")
nn.out.link(pointsOut.input)


# start server and wait for foxglove connection
async def main():
    class Listener(FoxgloveServerListener):
        def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("First client subscribed to", channel_id)

        def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("Last client unsubscribed from", channel_id)

    async with FoxgloveServer("0.0.0.0", 8765, "DepthAI server") as server:
        server.set_listener(Listener())

        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html
        if not args.disable_pcl:
            pointCloudChanel = await server.add_channel(
                {
                    "topic": "pointCloud",
                    "encoding": "json",
                    "schemaName": "ros.sensor_msgs.PointCloud2",
                    "schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "header": {
                                    "type": "object",
                                    "properties": {
                                        "seq": {"type": "integer"},
                                        "stamp": {
                                            "type": "object",
                                            "properties": {
                                                "sec": {"type": "integer"},
                                                "nsec": {"type": "integer"},
                                            },
                                        },
                                        "frame_id": {"type": "string"}
                                    },
                                },
                                "height": {"type": "integer"},
                                "width": {"type": "integer"},
                                "fields": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "offset": {"type": "integer"},
                                            "datatype": {"type": "integer"},
                                            "count": {"type": "integer"}
                                        }
                                    },
                                },
                                "is_bigendian": {"type": "boolean"},
                                "point_step": {"type": "integer"},
                                "row_step": {"type": "integer"},
                                "data": {"type": "string", "contentEncoding": "base64"},
                                "is_dense": {"type": "boolean"}
                            },
                        },
                    ),
                }
            )
        if not args.disable_color:
            colorChannel = await server.add_channel(
                {
                    "topic": "colorImage",
                    "encoding": "json",
                    "schemaName": "ros.sensor_msgs.CompressedImage",
                    "schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "header": {
                                    "type": "object",
                                    "properties": {
                                        "stamp": {
                                            "type": "object",
                                            "properties": {
                                                "sec": {"type": "integer"},
                                                "nsec": {"type": "integer"},
                                            },
                                        },
                                    },
                                },
                                "format": {"type": "string"},
                                "data": {"type": "string", "contentEncoding": "base64"},
                            },
                        },
                    ),
                }
            )
        if args.left:
            leftChannel = await server.add_channel(
                {
                    "topic": "leftImage",
                    "encoding": "json",
                    "schemaName": "ros.sensor_msgs.CompressedImage",
                    "schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "header": {
                                    "type": "object",
                                    "properties": {
                                        "stamp": {
                                            "type": "object",
                                            "properties": {
                                                "sec": {"type": "integer"},
                                                "nsec": {"type": "integer"},
                                            },
                                        },
                                    },
                                },
                                "format": {"type": "string"},
                                "data": {"type": "string", "contentEncoding": "base64"},
                            },
                        },
                    ),
                }
            )
        if args.right:
            rightChannel = await server.add_channel(
                {
                    "topic": "rightImage",
                    "encoding": "json",
                    "schemaName": "ros.sensor_msgs.CompressedImage",
                    "schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "header": {
                                    "type": "object",
                                    "properties": {
                                        "stamp": {
                                            "type": "object",
                                            "properties": {
                                                "sec": {"type": "integer"},
                                                "nsec": {"type": "integer"},
                                            },
                                        },
                                    },
                                },
                                "format": {"type": "string"},
                                "data": {"type": "string", "contentEncoding": "base64"},
                            },
                        },
                    ),
                }
            )

        seq = 0
        with dai.Device(pipeline) as device:
            print("Opening device")
            if not args.disable_pcl:
                calibData = device.readCalibration()
                M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT,
                                                        dai.Size2f(resolution[0], resolution[1]),
                                                        )

                # Creater xyz data and send it to the device - to the pointcloud generation model (NeuralNetwork node)
                xyz = create_xyz(resolution[0], resolution[1], np.array(M_right).reshape(3, 3))
                matrix = np.array([xyz], dtype=np.float16).view(np.int8)
                buff = dai.Buffer()
                buff.setData(matrix)
                device.getInputQueue("xyz_in").send(buff)
                queue = device.getOutputQueue("pcl", maxSize=8, blocking=False)
            if not args.disable_color:
                qRgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
            if args.left:
                qLeft = device.getOutputQueue("left", maxSize=1, blocking=False)
            if args.right:
                qRight = device.getOutputQueue("right", maxSize=1, blocking=False)

            while True:
                tmpTime = time.time_ns()
                sec = math.trunc(tmpTime / 1e9)
                nsec = tmpTime - sec

                await asyncio.sleep(0.1)

                if not args.disable_pcl:
                    pcl_data = np.array(queue.get().getFirstLayerFp16()).reshape(1, 3, resolution[1], resolution[0])
                    pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0

                    pcl_converter = PointCloudVisualizer()
                    pcl_converter.visualize_pcl(pcl_data, downsample=downsample_pcl)
                    pcl_data = pcl_converter.pcl.points

                    buf = bytes()
                    for point in pcl_data:
                        buf += struct.pack('f', float(point[0]))
                        buf += struct.pack('f', float(point[1]))
                        buf += struct.pack('f', float(point[2]))
                    # data needs to be encoded in base64
                    data = base64.b64encode(buf).decode("ascii")

                    # data is sent with json (data must be in above schema order)
                    await server.send_message(
                        pointCloudChanel,
                        time.time_ns(),
                        json.dumps(
                            {
                                "header": {
                                    "seq": seq,
                                    "stamp": {"sec": sec, "nsec": nsec},
                                    "frame_id": "front"
                                },
                                "height": 1,
                                "width": len(pcl_data),
                                "fields": [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                                           {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                                           {"name": "z", "offset": 8, "datatype": 7, "count": 1}],
                                "is_bigendian": False,
                                "point_step": 12,
                                "row_step": 12 * len(pcl_data),
                                "data": data,
                                "is_dense": True
                            }
                        ).encode("utf8"),
                    )
                    seq += 1

                if not args.disable_color:
                    if qRgb.has():
                        img = qRgb.get().getCvFrame()
                        is_success, im_buf_arr = cv2.imencode(".jpg", img)

                        # read from .jpeg format to buffer of bytes
                        byte_im = im_buf_arr.tobytes()

                        # data must be encoded in base64
                        data = base64.b64encode(byte_im).decode("ascii")

                        # data is sent with json (data must be in above schema order)
                        await server.send_message(
                            colorChannel,
                            time.time_ns(),
                            json.dumps(
                                {
                                    "header": {"stamp": {"sec": sec, "nsec": nsec}},
                                    "format": "jpeg",
                                    "data": data,
                                }
                            ).encode("utf8"),
                        )

                if args.left:
                    img = qLeft.get().getCvFrame()
                    is_success, im_buf_arr = cv2.imencode(".jpg", img)

                    # read from .jpeg format to buffer of bytes
                    byte_im = im_buf_arr.tobytes()

                    # data must be encoded in base64
                    data = base64.b64encode(byte_im).decode("ascii")

                    # data is sent with json (data must be in above schema order)
                    await server.send_message(
                        leftChannel,
                        time.time_ns(),
                        json.dumps(
                            {
                                "header": {"stamp": {"sec": sec, "nsec": nsec}},
                                "format": "jpeg",
                                "data": data,
                            }
                        ).encode("utf8"),
                    )

                if args.right:
                    if qRight.has():
                        img = qRight.get().getCvFrame()
                        is_success, im_buf_arr = cv2.imencode(".jpg", img)

                        # read from .jpeg format to buffer of bytes
                        byte_im = im_buf_arr.tobytes()

                        # data must be encoded in base64
                        data = base64.b64encode(byte_im).decode("ascii")

                        # data is sent with json (data must be in above schema order)
                        await server.send_message(
                            rightChannel,
                            time.time_ns(),
                            json.dumps(
                                {
                                    "header": {"stamp": {"sec": sec, "nsec": nsec}},
                                    "format": "jpeg",
                                    "data": data,
                                }
                            ).encode("utf8"),
                        )

                if cv2.waitKey(1) == "q":
                    break


if __name__ == "__main__":
    run_cancellable(main())
