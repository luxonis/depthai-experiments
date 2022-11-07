import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode, TextPosition

reid_results = []


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def callback(packet, visualizer, **kwargs):
    for i, (detection, recognition) in enumerate(zip(packet.detections, packet.nnData)):
        reid = recognition.getFirstLayerFp16()

        for j, vector in enumerate(reid_results):
            dist = cos_dist(reid, vector)
            if dist > 0.7:
                reid_results[j] = np.array(reid)
                break
        else:
            reid_results.append(np.array(reid))

        visualizer.add_text(f'Person {i}',
                            bbox=(*detection.top_left, *detection.bottom_right),
                            position=TextPosition.TOP_LEFT)

    visualizer.draw(packet.frame)
    cv2.imshow('Pedestrian detection', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    person_detection = oak.create_nn('person-detection-retail-0013', color, type='mobilenet')
    person_detection.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.CROP)

    person_recognition = oak.create_nn('person-reidentification-retail-0288', person_detection)

    visualizer = oak.visualize(person_recognition, callback=callback)
    visualizer.detections(hide_label=True)

    oak.start(blocking=True)
