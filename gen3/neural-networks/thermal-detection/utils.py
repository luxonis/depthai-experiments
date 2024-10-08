import cv2
import depthai as dai
import numpy as np
from fps import FPS

# Labels: 0 for vehicle and 1 for person
labels = ["vehicle", "person"]

class BaseDetection(dai.node.HostNode):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_THICKNESS = 2

    def __init__(self, detected_labels=None) -> None:
        """
        Initialize the base detection class with an optional list of labels to detect.
        """
        self.fps_counter = FPS()
        self.detected_labels = detected_labels if detected_labels is not None else []
        super().__init__()

    def build(self, img_frame: dai.Node.Output, detections: dai.Node.Output) -> "BaseDetection":
        self.link_args(img_frame, detections)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, img_frame, detections: dai.ImgDetections) -> None:
        assert isinstance(img_frame, dai.ImgFrame)
        frame: np.ndarray = img_frame.getCvFrame()

        if detections is not None:
            for detection in detections.detections:
                label = labels[detection.label]
                if label in self.detected_labels:  # Only process labels specified in `detected_labels`
                    tl, br = self.denormalize(detection.xmin, detection.ymin, detection.xmax, detection.ymax, frame.shape)
                    self.draw_bbox(frame, tl, br, (0, 255, 0), 1, label)

        text = f'FPS: {self.fps_counter.fps():.1f}'
        self.fps_counter.next_iter()

        x, y = self.get_text_relative_position(text, frame.shape)
        self.draw_text((x, y), text, frame)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()

    def denormalize(self, xmin, ymin, xmax, ymax, frame_shape) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (int(frame_shape[1] * xmin), int(frame_shape[0] * ymin)),
            (int(frame_shape[1] * xmax), int(frame_shape[0] * ymax))
        )

    def draw_bbox(self,
                  img: np.ndarray,
                  pt1: tuple[int, int],
                  pt2: tuple[int, int],
                  color: tuple[int, int, int],
                  thickness: int,
                  label: str = '',
                  alpha: float = 0.15
                  ) -> None:
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw bounding box lines and rounded corners
        line_width = np.abs(x2 - x1)
        line_height = np.abs(y2 - y1)
        cv2.line(img, (x1, y1), (x1 + line_width, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + line_height), color, thickness)
        cv2.ellipse(img, (x1, y1), (0, 0), 180, 0, 90, color, thickness)

        cv2.line(img, (x2, y1), (x2 - line_width, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_height), color, thickness)
        cv2.ellipse(img, (x2, y1), (0, 0), 270, 0, 90, color, thickness)

        cv2.line(img, (x1, y2), (x1 + line_width, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - line_height), color, thickness)
        cv2.ellipse(img, (x1, y2), (0, 0), 90, 0, 90, color, thickness)

        cv2.line(img, (x2, y2), (x2 - line_width, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - line_height), color, thickness)
        cv2.ellipse(img, (x2, y2), (0, 0), 0, 0, 90, color, thickness)

        if alpha > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, pt1, pt2, color, thickness=cv2.FILLED)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw the label above the bounding box
        if label:
            label_size, _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            label_pt1 = (pt1[0], pt1[1] - label_size[1] - 5)  # Position above the top-left corner
            label_pt2 = (pt1[0] + label_size[0], pt1[1])

            # Background for the label text
            cv2.rectangle(img, label_pt1, label_pt2, color, cv2.FILLED)

            # Put the label text on the image
            cv2.putText(img, label, (pt1[0], pt1[1] - 5), self.FONT, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS)

    def get_text_relative_position(self, text: str, frame_shape, padding: int = 10) -> tuple[int, int]:
        bbox = (0.0, 0.0, 1.0, 1.0)
        tl, br = self.denormalize(*bbox, frame_shape)

        bbox_arr = (*tl, *br)
        text_width, text_height = 0, 0
        for text_line in text.splitlines():
            text_size = cv2.getTextSize(text=text_line, fontFace=self.FONT, fontScale=self.FONT_SCALE, thickness=self.FONT_THICKNESS)[0]
            text_width = max(text_width, text_size[0])
            text_height += text_size[1]

        x, y = bbox_arr[0] + padding, bbox_arr[1] + text_height + padding
        return x, y

    def draw_text(self, coords: tuple[int, int], text: str, frame: np.ndarray) -> None:
        font_scale = self.get_text_scale(frame.shape[:2])
        font_thickness = max(1, int(font_scale * 2))
        dx, dy = cv2.getTextSize(text, self.FONT, font_scale, font_thickness)[0]
        dy += 10

        for line in text.splitlines():
            y = coords[1]
            cv2.putText(img=frame, text=line, org=coords, fontFace=self.FONT, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness + 1, lineType=cv2.LINE_AA)
            cv2.putText(img=frame, text=line, org=coords, fontFace=self.FONT, fontScale=font_scale, color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
            coords = (coords[0], y + dy)

    def get_text_scale(self, frame_shape) -> float:
        return min(1.0, min(frame_shape) / 1000)


# Class to detect both vehicles and persons
class VehiclePersonDetection(BaseDetection):
    def __init__(self) -> None:
        super().__init__(detected_labels=["vehicle", "person"])

# Class to detect only persons
class PersonDetection(BaseDetection):
    def __init__(self) -> None:
        super().__init__(detected_labels=["person"])
