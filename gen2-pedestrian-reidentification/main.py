#!/usr/bin/env python3

from depthai_sdk import OakCamera, AspectRatioResizeMode, TwoStagePacket, TextPosition
import depthai as dai
import numpy as np
import cv2

class PedestrianReId:
    def __init__(self) -> None:
        self.results = []

    def cosine_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def new_result(self, vector_result) -> int:
        vector_result = np.array(vector_result)
        for i, vector in enumerate(self.results):
            dist = self.cosine_dist(vector, vector_result)
            if dist > 0.7:
                self.results[i] = vector_result
                return i
        else:
            self.results.append(vector_result)
            return len(self.results) - 1


with OakCamera(replay='people-construction-vest-01') as oak:
    color = oak.create_camera('color')
    person_det = oak.create_nn('person-detection-retail-0013', color)
    # Passthrough is enabled for debugging purposes
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    person_det.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.CROP)

    oak.visualize([person_det, person_det.out.passthrough])
    oak.start(blocking=True)
