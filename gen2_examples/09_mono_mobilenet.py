from pathlib import Path
import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam_left = pipeline.createMonoCamera()
cam_left.setCamId(1)
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)

detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/mobilenet-ssd.blob')).resolve().absolute()))
cam_left.out.link(detection_nn.input)

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")
cam_left.out.link(xout_left.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q_left = device.getOutputQueue("left")
q_nn = device.getOutputQueue("nn")

frame = None
bboxes = []


def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


while True:
    in_left = q_left.tryGet()
    in_nn = q_nn.tryGet()

    if in_left is not None:
        shape = (in_left.getHeight(), in_left.getWidth())
        frame = in_left.getData().reshape(shape).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

    if frame is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
