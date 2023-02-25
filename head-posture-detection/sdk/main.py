import cv2
import numpy as np
from depthai import NNData

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import TextPosition

MIN_THRESHOLD = 0.15


def decode_recognition(nn_data: NNData):
    yaw = nn_data.getLayerFp16('angle_y_fc')[0]
    pitch = nn_data.getLayerFp16('angle_p_fc')[0]
    roll = nn_data.getLayerFp16('angle_r_fc')[0]

    vals = np.array([abs(pitch), abs(yaw), abs(roll)])
    max_index = np.argmax(vals)

    if vals[max_index] < MIN_THRESHOLD:
        return None

    txt = ''
    if max_index == 0:
        if pitch > 0:
            txt = 'Look down'
        else:
            txt = 'Look up'
    elif max_index == 1:
        if yaw > 0:
            txt = 'Turn right'
        else:
            txt = 'Turn left'
    elif max_index == 2:
        if roll > 0:
            txt = 'Tilt right'
        else:
            txt = 'Tilt left'

    return txt, (yaw, pitch, roll)


def callback(packet):
    visualizer = packet.visualizer
    for det, rec in zip(packet.detections, packet.nnData):
        bbox = (*det.top_left, *det.bottom_right)

        if rec is None:
            continue

        txt, (yaw, pitch, roll) = rec

        visualizer.add_text('Pitch: {:.0f}\nYaw: {:.0f}\nRoll: {:.0f}'.format(pitch, yaw, roll),
                            bbox=bbox, position=TextPosition.BOTTOM_LEFT)

        visualizer.add_text(f'{txt}', bbox=bbox, position=TextPosition.TOP_LEFT)

    frame = visualizer.draw(packet.frame)
    cv2.imshow('Head posture detection', frame)


with OakCamera() as oak:
    camera = oak.create_camera('rgb', resolution='1080p', fps=30)

    face_nn = oak.create_nn('face-detection-retail-0004', camera)
    face_nn.config_nn(resize_mode='stretch')

    recognition_nn = oak.create_nn('head-pose-estimation-adas-0001', face_nn, decode_fn=decode_recognition)

    visualizer = oak.visualize([recognition_nn.out.main], callback=callback, fps=True)
    visualizer.detections(hide_label=True)

    oak.start(blocking=True)
