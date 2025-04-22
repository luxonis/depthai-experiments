#!/usr/bin/env python3

import depthai as dai
import threading
import cv2
from utility import filter_internal_cameras, run_pipeline
from typing import List, Callable

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

labelMap = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class OpencvManager:
    def __init__(self, keys: List[int]):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.frames: dict[int, dai.ImgFrame] = self._init_dictionary(keys)
        self.detections: dict[int, dai.ImgDetections] = self._init_dictionary(keys)

    def run(self) -> None:
        while True:
            self.newFrameEvent.wait()
            for dx_id in self.detections.keys():
                if self.detections[dx_id] is not None:
                    frame = self.frames[dx_id].getCvFrame()
                    for detection in self.detections[dx_id].detections:
                        ymin = int(300 * detection.ymin)
                        xmin = int(300 * detection.xmin)
                        cv2.putText(
                            frame,
                            labelMap[detection.label],
                            (xmin + 10, ymin + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                        )
                        cv2.putText(
                            frame,
                            f"{int(detection.confidence * 100)}%",
                            (xmin + 10, ymin + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                        )
                        cv2.rectangle(
                            frame,
                            (xmin, ymin),
                            (int(300 * detection.xmax), int(300 * detection.ymax)),
                            (255, 255, 255),
                            2,
                        )

                    cv2.imshow(f"Preview - {dx_id}", frame)

                    if cv2.waitKey(1) == ord("q"):
                        return

    def set_outputs(
        self, frame: dai.ImgFrame, detections: dai.ImgDetections, dx_id: int
    ) -> None:
        with self.lock:
            self.frames[dx_id] = frame
            self.detections[dx_id] = detections
            self.newFrameEvent.set()

    def _init_dictionary(self, keys: List[int]) -> dict:
        dic = dict()
        for key in keys:
            dic[key] = None
        return dic


class Display(dai.node.HostNode):
    def __init__(self, callback: Callable, dx_id: int) -> None:
        super().__init__()
        self.callback = callback
        self.dx_id = dx_id

    def build(self, cam_out: dai.Node.Output, det_out: dai.Node.Output) -> "Display":
        self.link_args(cam_out, det_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, in_frame: dai.ImgFrame, in_det: dai.ImgDetections) -> None:
        self.callback(in_frame, in_det, self.dx_id)


def getPipeline(dev: dai.Device, callback: Callable) -> dai.Pipeline:
    pipeline = dai.Pipeline(dev)

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_preview = cam_rgb.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p)

    detector: dai.node.DetectionNetwork = pipeline.create(dai.node.DetectionNetwork)
    detector.setConfidenceThreshold(0.5)
    detector.setNNArchive(nn_archive)
    rgb_preview.link(detector.input)

    pipeline.create(Display, callback, dev.getMxId()).build(
        cam_out=rgb_preview, det_out=detector.out
    )

    return pipeline


def pair_device_with_pipeline(
    device_info: dai.DeviceInfo, pipelines: List[dai.Pipeline], callback: Callable
) -> None:
    device = dai.Device(device_info)

    mxid = device_info.getMxId()
    print("=== Connected to " + mxid)

    pipeline: dai.Pipeline = getPipeline(device, callback)
    pipelines.append(pipeline)


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())
if len(devices) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(devices), "devices")

pipelines: List[dai.Pipeline] = []
threads: List[threading.Thread] = []
manager = OpencvManager([device.getMxId() for device in devices])

for dev in devices:
    pair_device_with_pipeline(dev, pipelines, manager.set_outputs)

for pipeline in pipelines:
    thread = threading.Thread(target=run_pipeline, args=(pipeline,))
    thread.start()
    threads.append(thread)

manager.run()

for pipeline in pipelines:
    pipeline.stop()

for t in threads:
    t.join()  # Wait for all threads to finish (to connect to devices)
