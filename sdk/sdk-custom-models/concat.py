import cv2
import depthai as dai
import numpy as np

from depthai_sdk import OakCamera, DetectionPacket


def get_frame(data: dai.NNData, shape):
    diff = np.array(data.getFirstLayerFp16()).reshape(shape)
    colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)


def callback(packet: DetectionPacket):
    edge_frame = get_frame(packet.img_detections, (3, 300, 300))

    cv2.imshow("Laplacian edge detection", edge_frame)
    cv2.imshow("Color", packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    raise NotImplementedError('TODO: Multiple NN inputs')

    nn = oak.create_nn('models/concat_openvino_2021.4_6shave.blob', color)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
