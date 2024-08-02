import cv2
import numpy as np

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self._bboxColors = np.random.random(size=(256, 3)) * 256
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.6, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.6, self.color, 1, self.line_type)
    def rectangle(self, frame, p1, p2, id):
        cv2.rectangle(frame, p1, p2, (0,0,0), 4)
        cv2.rectangle(frame, p1, p2, self._bboxColors[id], 2)

class TitleHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self._bboxColors = np.random.random(size=(256, 3)) * 256
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.2, self.bg_color, 6, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.2, self.color, 2, self.line_type)
    def rectangle(self, frame, p1, p2, id):
        cv2.rectangle(frame, p1, p2, self._bboxColors[id], 3)