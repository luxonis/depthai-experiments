#!/usr/bin/env python3

import depthai as dai
import threading
import cv2
import blobconverter


class OpencvManager:
    def __init__(self, keys : list[int]):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.frames : dict[int, dai.ImgFrame] = self._init_dictionary(keys)
        self.detections : dict[int, dai.ImgDetections] = self._init_dictionary(keys)


    def run(self) -> None:
        while True:
            self.newFrameEvent.wait()
            for dx_id in self.detections.keys():
                if self.detections[dx_id] is not None:

                    frame = self.frames[dx_id].getCvFrame()
                    for detection in self.detections[dx_id].detections:
                        ymin = int(300*detection.ymin)
                        xmin = int(300*detection.xmin)
                        cv2.putText(frame, labelMap[detection.label], (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (xmin + 10, ymin + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                        cv2.rectangle(frame, (xmin, ymin), (int(300*detection.xmax), int(300*detection.ymax)), (255,255,255), 2)

                    cv2.imshow(f"Preview - {dx_id}", frame)

                    if cv2.waitKey(1) == ord('q'):
                        return
    

    def set_outputs(self, frame : dai.ImgFrame, detections : dai.ImgDetections, dx_id : int) -> None:
        with self.lock:
            self.frames[dx_id] = frame
            self.detections[dx_id] = detections
            self.newFrameEvent.set()

    
    def _init_dictionary(self, keys : list[int]) -> dict:
        dic = dict()
        for key in keys:
            dic[key] = None
        return dic


class Display(dai.node.HostNode):
    def __init__(self, callback : callable, dx_id : int) -> None:
        super().__init__()
        self.callback = callback
        self.dx_id = dx_id  


    def build(self, cam_out : dai.Node.Output, det_out : dai.Node.Output) -> "Display":
        self.link_args(cam_out, det_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, in_frame : dai.ImgFrame, in_det : dai.ImgDetections) -> None:
        self.callback(in_frame, in_det, self.dx_id)


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def run_pipeline(pipeline : dai.Pipeline) -> None:
    pipeline.run()


def getPipeline(dev : dai.Device, callback : callable) -> dai.Pipeline:
    pipeline = dai.Pipeline(dev)

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)  
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    detector : dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
    detector.setConfidenceThreshold(0.5)
    detector.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
    cam_rgb.preview.link(detector.input)

    pipeline.create(Display, callback, dev.getMxId()).build(
        cam_out=cam_rgb.preview,
        det_out=detector.out
    )

    return pipeline

def pair_device_with_pipeline(device_info : dai.DeviceInfo, pipelines : list[dai.Pipeline], callback : callable) -> None:
    device = dai.Device(device_info)

    mxid = device_info.getMxId()
    print("=== Connected to " + mxid)
    
    pipeline : dai.Pipeline = getPipeline(device, callback)
    pipelines.append(pipeline)


devices = filterInternalCameras(dai.Device.getAllAvailableDevices())
if len(devices) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(devices), "devices")

pipelines : list[dai.Pipeline] = []
threads : list[threading.Thread] = []
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
    t.join() # Wait for all threads to finish (to connect to devices)
