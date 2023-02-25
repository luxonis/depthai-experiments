import cv2
import numpy as np

from depthai_sdk import OakCamera

NN_WIDTH, NN_HEIGHT = 256, 256
N_CLASSES = 21


def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(NN_WIDTH, NN_HEIGHT)

    # scale to [0 ... 255] and apply colormap
    output = np.array(output) * (255 / N_CLASSES)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    # reset the color of 0 class
    output_colors[output == 0] = [0, 0, 0]

    return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.4, 0)


def callback(packet):
    frame = packet.frame
    nn_data = packet.img_detections

    # reshape to numpy array
    layer1 = np.array(nn_data.getFirstLayerInt32()).reshape(NN_WIDTH, NN_HEIGHT)

    found_classes = np.unique(layer1)
    output_colors = decode_deeplabv3p(layer1)
    output_colors = cv2.resize(output_colors, (frame.shape[1], frame.shape[0]))

    frame = show_deeplabv3p(output_colors, frame)
    cv2.putText(frame, "Found classes {}".format(found_classes), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (255, 255, 255))

    cv2.imshow("DeepLabV3 segmentation", frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn = oak.create_nn('models/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob', color)
    nn.config_nn(resize_mode='stretch')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
