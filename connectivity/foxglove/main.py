import argparse as argparse
import asyncio
import math
import time

import depthai as dai
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer
from utils.arguments import initialize_argparser
from utils.foxglove_utils import (
    Listener,
    create_channels,
    process_pointcloud,
    send_frame,
    send_pointcloud,
)

_, args = initialize_argparser()

resolution = (640, 400)

# To speed up visualization downsample the pointcloud before operating on it and visualizing
downsample_pcl = True


async def main():
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        color_q = None
        left_q = None
        right_q = None
        pcl_q = None
        if not args.no_color:
            color_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A
            )
            color_out = color_cam.requestOutput(
                resolution, type=dai.ImgFrame.Type.NV12, fps=args.fps_limit
            )
            color_q = color_out.createOutputQueue(blocking=False, maxSize=1)

        if args.pointcloud or args.left:
            left_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B
            )
            left_out = left_cam.requestOutput(
                resolution, type=dai.ImgFrame.Type.NV12, fps=args.fps_limit
            )

            if args.left:
                left_q = left_out.createOutputQueue(blocking=False, maxSize=1)

        if args.pointcloud or args.right:
            right_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C
            )
            right_out = right_cam.requestOutput(
                resolution, type=dai.ImgFrame.Type.NV12, fps=args.fps_limit
            )

            if args.right:
                right_q = right_out.createOutputQueue(blocking=False, maxSize=1)

        if args.pointcloud:
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left_out, right=right_out
            )
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(False)
            stereo.setSubpixel(True)
            stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

            # In-place post-processing configuration for a stereo depth node
            # The best combo of filters is application specific. Hard to say there is a one size fits all.
            # They also are not free. Even though they happen on device, you pay a penalty in fps.
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

            if not args.no_color:
                img_align = pipeline.create(dai.node.ImageAlign)
                color_out.link(img_align.inputAlignTo)
                stereo.depth.link(img_align.input)
                depth_out = img_align.outputAligned
            else:
                depth_out = stereo.depth

            point_cloud = pipeline.create(dai.node.PointCloud)
            depth_out.link(point_cloud.inputDepth)
            pcl_q = point_cloud.outputPointCloud.createOutputQueue(
                blocking=False, maxSize=8
            )
        else:
            pcl_q = None

        print("Pipeline created.")
        pipeline.start()

        # Start server and wait for foxglove connection
        async with FoxgloveServer("0.0.0.0", 8765, "DepthAI server") as server:
            server.set_listener(Listener())

            (
                color_channel,
                pointcloud_channel,
                left_channel,
                right_channel,
            ) = await create_channels(
                server, not args.no_color, args.pointcloud, args.left, args.right
            )

            while pipeline.isRunning():
                tmpTime = time.time_ns()
                sec = math.trunc(tmpTime / 1e9)
                nsec = tmpTime - sec

                await asyncio.sleep(0.01)

                if color_q is not None and color_q.has():
                    frame = color_q.get().getCvFrame()
                    await send_frame(server, frame, sec, nsec, color_channel)
                if left_q is not None and left_q.has():
                    frame = left_q.get().getCvFrame()
                    await send_frame(server, frame, sec, nsec, left_channel)
                if right_q is not None and right_q.has():
                    frame = right_q.get().getCvFrame()
                    await send_frame(server, frame, sec, nsec, right_channel)
                if pcl_q is not None and pcl_q.has():
                    pcl_data = pcl_q.get()
                    pcl_processed = process_pointcloud(
                        pcl_data.getPoints(), downsample_pcl
                    )
                    await send_pointcloud(
                        server, pcl_processed, sec, nsec, pointcloud_channel
                    )


if __name__ == "__main__":
    run_cancellable(main())
