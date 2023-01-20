import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode, TwoStagePacket, Visualizer, TextPosition


def callback(packet: TwoStagePacket, visualizer: Visualizer):
    for det, rec in zip(packet.detections, packet.nnData):
        has_mask = np.argmax(rec.getFirstLayerFp16())
        mask_str = "Mask" if has_mask else "No mask"

        visualizer.add_text(f'{mask_str}',
                            bbox=(*det.top_left, *det.bottom_right),
                            position=TextPosition.BOTTOM_RIGHT)

    frame = visualizer.draw(packet.frame)
    cv2.imshow('Mask detection', frame)


with OakCamera() as oak:
    camera = oak.create_camera('rgb', resolution='1080p', fps=30)

    face_nn = oak.create_nn('face-detection-retail-0004', camera)
    face_nn.config_nn(aspect_ratio_resize_mode=AspectRatioResizeMode.CROP)

    recognition_nn = oak.create_nn('../models/sbd_mask_classification_224x224.blob', face_nn)

    oak.visualize([recognition_nn.out.main], callback=callback, fps=True)
    oak.start(blocking=True)
