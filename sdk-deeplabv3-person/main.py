import blobconverter
import cv2
import numpy as np
from depthai import NNData

from depthai_sdk import OakCamera, DetectionPacket, AspectRatioResizeMode

NN_WIDTH, NN_HEIGHT = 513, 513
N_CLASSES = 21


def decode_deeplabv3p(output_tensor):
    class_colors = [[0, 0, 0], [0, 255, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(NN_WIDTH, NN_HEIGHT)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.2, 0)


def callback(packet: DetectionPacket):
    frame = packet.frame
    nn_data: NNData = packet.img_detections

    layer1 = np.array(nn_data.getFirstLayerInt32()).reshape(NN_WIDTH, NN_HEIGHT)

    output_colors = decode_deeplabv3p(layer1)
    output_colors = cv2.resize(output_colors, (frame.shape[1], frame.shape[0]))

    frame = show_deeplabv3p(output_colors, frame)
    cv2.imshow("DeepLabV3 person segmentation", frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    nn_path = blobconverter.from_zoo(name=f"deeplab_v3_mnv2_513x513", zoo_type="depthai", shaves=6)
    nn = oak.create_nn(nn_path, color)
    nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.STRETCH)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
