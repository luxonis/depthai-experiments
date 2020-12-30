from pathlib import Path

import cv2
import numpy as np
import depthai
import east

pipeline = depthai.Pipeline()

colorCam = pipeline.createColorCamera()
colorCam.setPreviewSize(256, 256)
colorCam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setCamId(0)

cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("preview")
colorCam.preview.link(cam_xout.input)

nn = pipeline.createNeuralNetwork()
nn.setBlobPath(str((Path(__file__).parent / Path('text-detection.blob')).resolve().absolute()))
colorCam.preview.link(nn.input)

nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("detections")
nn.out.link(nn_xout.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


q_prev = device.getOutputQueue("preview")
q_det = device.getOutputQueue("detections")

frame = None
points = []

while True:
    in_prev = q_prev.tryGet()
    in_det = q_det.tryGet()

    if in_prev is not None:
        shape = (3, in_prev.getHeight(), in_prev.getWidth())
        frame = in_prev.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_det is not None:
        scores, geom1, geom2 = to_tensor_result(in_det).values()
        scores = np.reshape(scores, (1, 1, 64, 64))
        geom1 = np.reshape(geom1, (1, 4, 64, 64))
        geom2 = np.reshape(geom2, (1, 1, 64, 64))

        bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
        boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
        points = [
            east.rotated_Rectangle(bbox, angle * -1)
            for (bbox, angle) in zip(boxes, angles)
        ]

    if frame is not None:
        for point_arr in points:
            cv2.polylines(frame, [point_arr], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

        cv2.imshow('preview', frame)
        if cv2.waitKey(1) == ord('q'):
            break
