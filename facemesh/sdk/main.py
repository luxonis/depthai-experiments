import cv2
import depthai as dai
import numpy as np

from depthai_sdk import OakCamera
from utils.effect import EffectRenderer2D

CONF_THRESH = 0.5
NN_WIDTH, NN_HEIGHT = 192, 192
PREVIEW_WIDTH, PREVIEW_HEIGHT = 416, 416
OVERLAY_IMAGE = 'mask/facepaint.png'

effect_rendered = EffectRenderer2D(OVERLAY_IMAGE)


def callback(packet):
    frame = packet.frame
    nn_data = packet.img_detections

    score = np.array(nn_data.getLayerFp16('conv2d_31')).reshape((1,))
    score = 1 / (1 + np.exp(-score[0]))  # sigmoid on score
    landmarks = np.array(nn_data.getLayerFp16('conv2d_21')).reshape((468, 3))

    if score > CONF_THRESH:
        # scale landmarks
        ldms = landmarks  # .copy()
        ar_diff = (PREVIEW_WIDTH / PREVIEW_HEIGHT) / (frame.shape[1] / frame.shape[0])

        ldms *= np.array([frame.shape[0] / NN_HEIGHT, frame.shape[0] / NN_HEIGHT, 1])
        ldms[:, 0] += (frame.shape[1] - frame.shape[1] * ar_diff) / 2

        # render frame
        target_frame = frame.copy()
        applied_effect = effect_rendered.render_effect(target_frame, ldms)

        # show landmarks on frame
        for ldm in ldms:
            col = (0, 0, int(ldm[2]) * 5 + 100)
            cv2.circle(frame, (int(ldm[0]), int(ldm[1])), 1, col, 1)

    else:
        applied_effect = frame

    cv2.imshow("FaceMesh", np.hstack([frame, applied_effect]))


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.config_color_camera(interleaved=False, color_order=dai.ColorCameraProperties.ColorOrder.RGB)

    nn = oak.create_nn('models/face_landmark_openvino_2021.4_6shave.blob', color)
    nn.config_nn(resize_mode='crop')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
