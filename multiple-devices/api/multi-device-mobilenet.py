#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import blobconverter

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# This can be customized to pass multiple parameters
def getPipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # For the demo, just set a larger RGB preview size for OAK-D
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    detector = pipeline.create(dai.node.MobileNetDetectionNetwork)
    detector.setConfidenceThreshold(0.5)
    detector.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
    cam_rgb.preview.link(detector.input)

    # Create output
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    detector.passthrough.link(xout_rgb.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    detector.out.link(xout_nn.input)

    return pipeline


# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()
    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")
    devices = {}

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

        # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
        print("=== Connected to " + device_info.getMxId())
        mxid = device.getMxId()
        cameras = device.getConnectedCameras()
        usb_speed = device.getUsbSpeed()

        # Get a customized pipeline based on identified device type
        pipeline = getPipeline()
        device.startPipeline(pipeline)

        # Output queue will be used to get the rgb frames from the output defined above
        devices[mxid] = {
            'rgb': device.getOutputQueue(name="rgb"),
            'nn': device.getOutputQueue(name="nn"),
        }


    while True:
        for mxid, q in devices.items():
            if q['nn'].has():
                dets = q['nn'].get().detections
                frame = q['rgb'].get().getCvFrame()

                for detection in dets:
                    ymin = int(300*detection.ymin)
                    xmin = int(300*detection.xmin)
                    cv2.putText(frame, labelMap[detection.label], (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (xmin + 10, ymin + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                    cv2.rectangle(frame, (xmin, ymin), (int(300*detection.xmax), int(300*detection.ymax)), (255,255,255), 2)
                # Show the frame

                cv2.imshow(f"Preview - {mxid}", frame)

        if cv2.waitKey(1) == ord('q'):
            break
