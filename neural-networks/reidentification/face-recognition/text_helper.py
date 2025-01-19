import cv2


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(
            frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type
        )
