import cv2
import depthai as dai
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode
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

    print(score)
    if score > CONF_THRESH:
        # scale landmarks
        ldms = landmarks  # .copy()
        ldms *= np.array([PREVIEW_WIDTH / NN_WIDTH, PREVIEW_HEIGHT / NN_HEIGHT, 1])

        # render frame
        target_frame = frame.copy()
        applied_effect = effect_rendered.render_effect(target_frame, ldms)
        cv2.imshow("Effect", applied_effect)

        # show landmarks on frame
        for ldm in ldms:
            col = (0, 0, int(ldm[2]) * 5 + 100)
            cv2.circle(frame, (int(ldm[0]), int(ldm[1])), 1, col, 1)

    else:
        applied_effect = frame

    cv2.imshow("Demo", np.hstack([frame, applied_effect]))


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.config_color_camera(interleaved=False, colorOrder=dai.ColorCameraProperties.ColorOrder.RGB)

    nn = oak.create_nn('models/face_landmark_openvino_2021.4_6shave.blob', color)
    nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.STRETCH)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
