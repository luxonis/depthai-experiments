#!/usr/bin/env python3

import cv2
import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.classes import TwoStagePacket
from depthai_sdk.visualize.configs import TextPosition


class PedestrianReId:
    def __init__(self) -> None:
        self.results = []

    def _cosine_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def new_result(self, vector_result) -> int:
        vector_result = np.array(vector_result)
        for i, vector in enumerate(self.results):
            dist = self._cosine_dist(vector, vector_result)
            if dist > 0.7:
                self.results[i] = vector_result
                return i
        else:
            self.results.append(vector_result)
            return len(self.results) - 1


def callback(packet: TwoStagePacket):
    visualizer = packet.visualizer
    for det, rec in zip(packet.detections, packet.nnData):
        reid_result = rec.getFirstLayerFp16()
        id = reid.new_result(reid_result)

        visualizer.add_text(f"ID: {id}",
                            bbox=(*det.top_left, *det.bottom_right),
                            position=TextPosition.MID)
    frame = visualizer.draw(packet.frame)
    cv2.imshow('Person reidentification', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=10)

    person_det = oak.create_nn('person-detection-retail-0013', color)
    person_det.node.setNumInferenceThreads(2)
    person_det.config_nn(resize_mode='crop')

    nn_reid = oak.create_nn('person-reidentification-retail-0288', input=person_det)
    nn_reid.node.setNumInferenceThreads(2)

    reid = PedestrianReId()
    results = []

    oak.visualize(nn_reid, callback=callback, fps=True)
    # oak.show_graph()
    oak.start(blocking=True)
