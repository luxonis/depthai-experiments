import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setCamId(0)
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q_rgb = device.getOutputQueue("rgb")

while True:
    in_rgb = q_rgb.get()
    shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
    frame_rgb = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    cv2.imshow("rgb", frame_rgb)

    if cv2.waitKey(1) == ord('q'):
        break
