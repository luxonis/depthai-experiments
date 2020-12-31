from pathlib import Path

import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

left = pipeline.createMonoCamera()
left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setCamId(1)

right = pipeline.createMonoCamera()
right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setCamId(2)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
# Note: the rectified streams are horizontally mirrored by default
depth.setOutputRectified(True)
depth.setRectifyEdgeFillColor(0) # Black, to better see the cutout
left.out.link(depth.left)
right.out.link(depth.right)

detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/mobilenet-ssd.blob')).resolve().absolute()))
depth.rectifiedLeft.link(detection_nn.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.disparity.link(xout_depth.input)

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("rect_left")
depth.rectifiedLeft.link(xout_left.input)

manip = pipeline.createImageManip()
manip.setResize(300, 300)
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manip.setFrameType(depthai.RawImgFrame.Type.BGR888p)
depth.rectifiedLeft.link(manip.inputImage)
manip.out.link(detection_nn.input)

xout_manip = pipeline.createXLinkOut()
xout_manip.setStreamName("manip")
manip.out.link(xout_manip.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q_left = device.getOutputQueue("rect_left")
q_manip = device.getOutputQueue("manip")
q_depth = device.getOutputQueue("depth")
q_nn = device.getOutputQueue("nn")

frame_left = None
frame_manip = None
frame_depth = None
bboxes = []


def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


while True:
    in_left = q_left.tryGet()
    in_manip = q_manip.tryGet()
    in_nn = q_nn.tryGet()
    in_depth = q_depth.tryGet()

    if in_left is not None:
        shape = (in_left.getHeight(), in_left.getWidth())
        frame_left = in_left.getData().reshape(shape).astype(np.uint8)
        frame_left = np.ascontiguousarray(frame_left)

    if in_manip is not None:
        shape = (3, in_manip.getHeight(), in_manip.getWidth())
        frame_manip = in_manip.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame_manip = np.ascontiguousarray(frame_manip)

    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

    if in_depth is not None:
        frame_depth = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
        frame_depth = np.ascontiguousarray(frame_depth)
        frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

    if frame_left is not None:
        cv2.imshow("rectif_left", frame_left)

    if frame_manip is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame_manip, raw_bbox)
            cv2.rectangle(frame_manip, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("manip", frame_manip)

    if frame_depth is not None:
        cv2.imshow("depth", frame_depth)

    if cv2.waitKey(1) == ord('q'):
        break
