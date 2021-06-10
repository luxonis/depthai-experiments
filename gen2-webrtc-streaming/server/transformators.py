import base64
import json
import traceback

import numpy as np
from aiortc import VideoStreamTrack
import cv2
from av import VideoFrame
import depthai as dai
import blobconverter


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
        print("TEST")
        print(options)
        self.frame = np.zeros((self.options.height, self.options.width, 3), np.uint8)
        self.frame[:] = (0, 0, 0)
        self.detections = []
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.createColorCamera()
        self.xoutRgb = self.pipeline.createXLinkOut()

        self.xoutRgb.setStreamName("rgb")

        # Properties
        self.camRgb.setPreviewSize(self.options.width, self.options.height)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Linking
        self.camRgb.preview.link(self.xoutRgb.input)
        self.nn = None
        if options.nn != "":
            self.nn = self.pipeline.createMobileNetDetectionNetwork()
            self.nn.setConfidenceThreshold(0.5)
            self.nn.setBlobPath(str(blobconverter.from_zoo(options.nn, shaves=6)))
            self.nn.setNumInferenceThreads(2)
            self.nn.input.setBlocking(False)
            self.nnOut = self.pipeline.createXLinkOut()
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
            cv2.putText(self.frame, f"LABEL {detection.label}", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        # Show the frame
        return self.frame

    def stop(self):
        super().stop()
        del self.device
