#!/usr/bin/env python3

import depthai as dai
import threading
import cv2
import blobconverter


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# This can be customized to pass multiple parameters
def getPipeline(dev, queues : list):
    # Start defining a pipeline
    pipeline = dai.Pipeline(dev)

    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera).build()
    # For the demo, just set a larger RGB preview size for OAK-D
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    detector = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    detector.setConfidenceThreshold(0.5)
    detector.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
    cam_rgb.preview.link(detector.input)

    queues.append(cam_rgb.preview.createOutputQueue())
    queues.append(detector.out.createOutputQueue())

    return pipeline

def worker(device_info, queues, pipelines):
    device = dai.Device(device_info)

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    mxid = device_info.getMxId()
    print("=== Connected to " + mxid)
    
    queues_out = []
    pipeline : dai.Pipeline = getPipeline(device, queues_out)
    pipelines.append(pipeline)

    queues[mxid] = {
        'rgb' : queues_out[0],
        'nn' : queues_out[1],
    }
    pipeline.start()



device_infos = filterInternalCameras(dai.Device.getAllAvailableDevices())
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")
device_queues = {}
threads = []
pipelines = []

for device_info in device_infos:
    thread = threading.Thread(target=worker, args=(device_info, device_queues, pipelines))
    thread.start()
    threads.append(thread)

for t in threads:
    t.join() # Wait for all threads to finish (to connect to devices)

while any(pipeline.isRunning() for pipeline in pipelines):
    for mxid, q in device_queues.items():
        if q['nn'].has():
            dets = q['nn'].tryGet().detections
            frame = q['rgb'].tryGet().getCvFrame()

            for detection in dets:
                ymin = int(300*detection.ymin)
                xmin = int(300*detection.xmin)
                cv2.putText(frame, labelMap[detection.label], (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (xmin + 10, ymin + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                cv2.rectangle(frame, (xmin, ymin), (int(300*detection.xmax), int(300*detection.ymax)), (255,255,255), 2)

            cv2.imshow(f"Preview - {mxid}", frame)

    if cv2.waitKey(1) == ord('q'):
        break
