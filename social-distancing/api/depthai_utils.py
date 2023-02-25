import logging
import uuid

import blobconverter
import cv2
import depthai as dai
from imutils.video import FPS

log = logging.getLogger(__name__)


class DepthAI:
    def create_pipeline(self, model_name):
        log.info("Creating DepthAI pipeline...")

        pipeline = dai.Pipeline()
        # pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        camRgb.preview.link(xoutRgb.input)
        xoutNN = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")

        # Properties
        camRgb.setPreviewSize(544, 320)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs
        stereo.setConfidenceThreshold(255)

        spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name=model_name, shaves=6))
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        log.info("Pipeline created.")
        return pipeline

    def __init__(self, model_name):
        self.pipeline = self.create_pipeline(model_name)
        self.detections = []

    def capture(self):
        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if not depth_enabled:
                raise RuntimeError(
                    "Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(
                        cams))
            device.startPipeline(self.pipeline)
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

            while True:
                frame = previewQueue.get().getCvFrame()
                inDet = detectionNNQueue.tryGet()

                if inDet is not None:
                    self.detections = inDet.detections

                bboxes = []
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in self.detections:
                    bboxes.append({
                        'id': uuid.uuid4(),
                        'label': detection.label,
                        'confidence': detection.confidence,
                        'x_min': int(detection.xmin * width),
                        'x_max': int(detection.xmax * width),
                        'y_min': int(detection.ymin * height),
                        'y_max': int(detection.ymax * height),
                        'depth_x': detection.spatialCoordinates.x / 1000,
                        'depth_y': detection.spatialCoordinates.y / 1000,
                        'depth_z': detection.spatialCoordinates.z / 1000,
                    })

                yield frame, bboxes

    def __del__(self):
        del self.pipeline


class DepthAIDebug(DepthAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = FPS()
        self.fps.start()

    def capture(self):
        for frame, detections in super().capture():
            self.fps.update()
            for detection in detections:
                cv2.rectangle(frame, (detection['x_min'], detection['y_min']), (detection['x_max'], detection['y_max']),
                              (0, 255, 0), 2)
                cv2.putText(frame, "x: {}".format(round(detection['depth_x'], 1)),
                            (detection['x_min'], detection['y_min'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "y: {}".format(round(detection['depth_y'], 1)),
                            (detection['x_min'], detection['y_min'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "z: {}".format(round(detection['depth_z'], 1)),
                            (detection['x_min'], detection['y_min'] + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "conf: {}".format(round(detection['confidence'], 1)),
                            (detection['x_min'], detection['y_min'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "label: {}".format(detection['label'], 1),
                            (detection['x_min'], detection['y_min'] + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            yield frame, detections

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
