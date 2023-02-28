import cv2
import numpy as np

from depthai_sdk import OakCamera

NN_WIDTH = 426
NN_HEIGHT = 240


def callback(packet):
    nn_data = packet.img_detections
    lay1 = np.array(nn_data.getFirstLayerFp16()).reshape((30, 52))  # density map output is 1/8 of input size
    count = np.sum(lay1)  # predicted count is the sum of the density map
    cv2.putText(packet.frame, f'Predicted count: {count:.2f}', (2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    output = np.array(lay1) * 255
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_VIRIDIS)
    output_colors = cv2.resize(output_colors, (packet.frame.shape[1], packet.frame.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    cv2.addWeighted(packet.frame, 1.0, output_colors, 0.5, 0, packet.frame)
    cv2.imshow('Crowd counting', packet.frame)


with OakCamera(replay='vids/vid1.mp4') as oak:
    color = oak.create_camera('color', fps=5)
    nn = oak.create_nn('models/vgg_openvino_2021.4_6shave.blob', color)

    oak.callback(nn.out.main, callback=callback)
    oak.start(blocking=True)
