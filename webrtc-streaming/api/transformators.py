import traceback

import blobconverter
import cv2
import depthai as dai
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, application, pc_id, options):
        super().__init__()  # don't forget this!
        self.dummy = False
        self.application = application
        self.pc_id = pc_id
        self.options = options
        self.frame = None

    async def get_frame(self):
        raise NotImplementedError()

    async def return_frame(self, frame):
        pts, time_base = await self.next_timestamp()
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    async def dummy_recv(self):
        frame = np.zeros((self.options.height, self.options.width, 3), np.uint8)
        y, x = frame.shape[0] / 2, frame.shape[1] / 2
        left, top, right, bottom = int(x - 50), int(y - 30), int(x + 50), int(y + 30)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "ERROR", (left, int((bottom + top) / 2 + 10)), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (255, 255, 255), 1)
        return await self.return_frame(frame)

    async def recv(self):
        if self.dummy:
            return await self.dummy_recv()
        try:
            frame = await self.get_frame()
            return await self.return_frame(frame)
        except:
            print(traceback.format_exc())
            print('Switching to dummy mode...')
            self.dummy = True
            return await self.dummy_recv()


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


class DepthAIVideoTransformTrack(VideoTransformTrack):
    def __init__(self, application, pc_id, options):
        super().__init__(application, pc_id, options)
        self.frame = np.zeros((self.options.height, self.options.width, 3), np.uint8)
        self.frame[:] = (0, 0, 0)
        self.detections = []
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")

        # Properties
        self.camRgb.setPreviewSize(self.options.width, self.options.height)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Linking
        self.camRgb.preview.link(self.xoutRgb.input)
        self.nn = None
        if options.nn != "":
            self.nn = self.pipeline.create(dai.node.MobileNetDetectionNetwork)
            self.nn.setConfidenceThreshold(0.5)
            self.nn.setBlobPath(blobconverter.from_zoo(options.nn, shaves=6))
            self.nn.setNumInferenceThreads(2)
            self.nn.input.setBlocking(False)
            self.nnOut = self.pipeline.create(dai.node.XLinkOut)
            self.nnOut.setStreamName("nn")
            self.camRgb.preview.link(self.nn.input)
            self.nn.out.link(self.nnOut.input)
        self.device = dai.Device(self.pipeline)
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        if self.nn is not None:
            self.qDet = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.device.startPipeline()

    async def get_frame(self):
        frame = self.qRgb.tryGet()
        if frame is not None:
            self.frame = frame.getCvFrame()
        if self.nn is not None:
            inDet = self.qDet.tryGet()
            if inDet is not None:
                self.detections = inDet.detections

        for detection in self.detections:
            bbox = frameNorm(self.frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(self.frame, f"LABEL {detection.label}", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        # Show the frame
        return self.frame

    def stop(self):
        super().stop()
        del self.device


class DepthAIDepthVideoTransformTrack(VideoTransformTrack):
    def __init__(self, application, pc_id, options):
        super().__init__(application, pc_id, options)
        self.frame = np.zeros((self.options.height, self.options.width, 3), np.uint8)
        self.frame[:] = (0, 0, 0)
        self.detections = []

        self.device = dai.Device()

        # Check if we have stereo cameras on the device
        cams = self.device.getConnectedCameras()
        depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
        if not depth_enabled:
            print("You are using camera that doesn't support stereo depth!")
            super().stop()
            del self.device
            return

        self.device.startPipeline(self.create_pipeline(options))
        self.qDepth = self.device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    def create_pipeline(self, options):
        self.pipeline = dai.Pipeline()

        self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
        self.monoRight = self.pipeline.create(dai.node.MonoCamera)
        self.depth = self.pipeline.create(dai.node.StereoDepth)
        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDepth.setStreamName("disparity")

        # Properties
        if options.mono_camera_resolution == 'THE_400_P':
            self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            self.frame = np.zeros((400, 640, 3), np.uint8)
        elif options.mono_camera_resolution == 'THE_720_P':
            self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            self.frame = np.zeros((720, 1280, 3), np.uint8)
        elif options.mono_camera_resolution == 'THE_800_P':
            self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            self.frame = np.zeros((800, 1280, 3), np.uint8)
        self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        self.depth.setConfidenceThreshold(200)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        if options.median_filter == 'MEDIAN_OFF':
            self.depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
        elif options.median_filter == 'KERNEL_3x3':
            self.depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_3x3)
        elif options.median_filter == 'KERNEL_5x5':
            self.depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_5x5)
        elif options.median_filter == 'KERNEL_7x7':
            self.depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        self.depth.setExtendedDisparity(options.extended_disparity)
        self.depth.setSubpixel(options.subpixel)

        # Linking
        self.monoLeft.out.link(self.depth.left)
        self.monoRight.out.link(self.depth.right)
        self.depth.disparity.link(self.xoutDepth.input)

        return self.pipeline

    async def get_frame(self):
        inDepth = self.qDepth.tryGet()
        if inDepth is not None:
            frame = inDepth.getFrame()
            frame = (frame * (255 / self.depth.getMaxDisparity())).astype(np.uint8)
            self.frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        return self.frame

    def stop(self):
        super().stop()
        del self.device
