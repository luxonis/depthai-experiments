import colorsys
from enum import Enum
import math
import cv2
import depthai as dai
import numpy as np
from dataclasses import dataclass
from detected_recognitions import DetectedRecognitions
from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class ColoredLabel:
    label: str
    color: tuple[int, int, int]


class TextPosition(Enum):
    TOP_LEFT = 0
    MID_LEFT = 1
    BOTTOM_LEFT = 2
    TOP_MID = 10
    MID_MID = 11
    BOTTOM_MID = 12
    TOP_RIGHT = 20
    MID_RIGHT = 21
    BOTTOM_RIGHT = 22


class DeepsortTracking(dai.node.HostNode):
    ALPHA = 0.15
    FONT_FACE = 0
    LINE_TYPE = cv2.LINE_AA
    FONT_SHADOW_COLOR = (0, 0, 0)
    FONT_COLOR = (255, 255, 255)
    TEXT_PADDING = 10

    def __init__(self) -> None:
        super().__init__()
        self._tracker = DeepSort(max_age=1000, nn_budget=None, embedder=None, nms_max_overlap=1.0, max_cosine_distance=0.2)


    def build(self, img_frames: dai.Node.Output, detected_recognitions: dai.Node.Output, labels: list[str]) -> "DeepsortTracking":
        self.link_args(img_frames, detected_recognitions)
        self.sendProcessingToPipeline(True)
        self._colored_labels = [ColoredLabel(label, color) for label, color in zip(labels, self.generate_colors(len(labels)))]
        return self
    

    def process(self, img_frame: dai.ImgFrame, detected_recognitions: dai.Buffer) -> None:
        frame = img_frame.getCvFrame()
        assert(isinstance(detected_recognitions, DetectedRecognitions))
        detections = detected_recognitions.detections.detections
        recognitions = detected_recognitions.nn_data
        
        if recognitions:
            object_tracks = self._tracker.iter(detections, recognitions, (640, 352))

            for track in object_tracks:
                if not track.is_confirmed() or \
                    track.time_since_update > 1 or \
                    track.detection_id >= len(detections) or \
                    track.detection_id < 0:
                    continue

                det = detections[track.detection_id]
                colored_label = self._colored_labels[det.label]
                self._draw_bb(det, frame, colored_label.color)
                self._draw_text(det, f'{colored_label.label} {det.confidence*100:.2f}%', frame, TextPosition.TOP_LEFT)
                self._draw_text(det, f'ID: {track.track_id}', frame, TextPosition.MID_MID)
        
        cv2.imshow("DeepSort tracker", frame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
        
    
    def _draw_bb(self, detection: dai.ImgDetection, frame: np.ndarray, color: tuple[int, int, int]) -> None:
        pt1, pt2 = self._denormalize_bounding_box(detection, frame.shape)
        self._draw_bbox(frame, pt1, pt2, color)


    def _denormalize_bounding_box(self, detection: dai.ImgDetection, frame_shape: tuple) -> tuple:
        return (
                (int(frame_shape[1] * detection.xmin), int(frame_shape[0] * detection.ymin)),
                (int(frame_shape[1] * detection.xmax), int(frame_shape[0] * detection.ymax))
            )
    

    def _draw_bbox(self, img: np.ndarray,
              pt1: tuple[int, int],
              pt2: tuple[int, int],
              color: tuple[int, int, int]
              ) -> None:
        x1, y1 = pt1
        x2, y2 = pt2

        line_width = np.abs(x2 - x1)
        line_height = np.abs(y2 - y1)

        cv2.line(img, (x1, y1), (x1 + line_width, y1), color)
        cv2.line(img, (x1, y1), (x1, y1 + line_height), color)

        cv2.line(img, (x2, y1), (x2 - line_width, y1), color)
        cv2.line(img, (x2, y1), (x2, y1 + line_height), color)

        cv2.line(img, (x1, y2), (x1 + line_width, y2), color)
        cv2.line(img, (x1, y2), (x1, y2 - line_height), color)

        cv2.line(img, (x2, y2), (x2 - line_width, y2), color)
        cv2.line(img, (x2, y2), (x2, y2 - line_height), color)

        if 0 < self.ALPHA:
            overlay = img.copy()

            thickness = -1
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])

            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[2], bbox[3])
            top_right = (bottom_right[0], top_left[1])
            bottom_left = (top_left[0], bottom_right[1])

            top_left_main_rect = (int(top_left[0]), int(top_left[1]))
            bottom_right_main_rect = (int(bottom_right[0]), int(bottom_right[1]))

            top_left_rect_left = (top_left[0], top_left[1])
            bottom_right_rect_left = (bottom_left[0], bottom_left[1])

            top_left_rect_right = (top_right[0], top_right[1])
            bottom_right_rect_right = (bottom_right[0], bottom_right[1])

            all_rects = [
                [top_left_main_rect, bottom_right_main_rect],
                [top_left_rect_left, bottom_right_rect_left],
                [top_left_rect_right, bottom_right_rect_right]
            ]

            [cv2.rectangle(overlay, pt1=rect[0], pt2=rect[1], color=color, thickness=thickness) for rect in all_rects]

            cv2.addWeighted(overlay, self.ALPHA, img, 1 - self.ALPHA, 0, img)


    def _draw_text(self, bbox: dai.ImgDetection, text: str, frame: np.ndarray, text_position: TextPosition) -> None:
        coords = self._get_relative_position(bbox, text, frame.shape, text_position)

        font_scale = self._get_text_scale(frame.shape, bbox)
        font_thickness = self._get_font_thickness(font_scale)

        dy = cv2.getTextSize(text, self.FONT_FACE, font_scale, font_thickness)[0][1] + 10

        for line in text.splitlines():
            y = coords[1]

            # Shadow text
            cv2.putText(img=frame,
                        text=line,
                        org=coords,
                        fontFace=self.FONT_FACE,
                        fontScale=font_scale,
                        color=self.FONT_SHADOW_COLOR,
                        thickness=font_thickness + 1,
                        lineType=self.LINE_TYPE)

            # Front text
            cv2.putText(img=frame,
                        text=line,
                        org=coords,
                        fontFace=self.FONT_FACE,
                        fontScale=font_scale,
                        color=self.FONT_COLOR,
                        thickness=font_thickness,
                        lineType=self.LINE_TYPE)

            coords = (coords[0], y + dy)


    def _get_relative_position(self, bbox: dai.ImgDetection, text: str, frame_shape: tuple[int, ...], position: TextPosition) -> tuple[int, int]:
        bbox_arr = self._to_tuple(bbox, frame_shape)

        font_scale = self._get_text_scale(frame_shape, bbox)
        font_face = self.FONT_FACE
        font_thickness = self._get_font_thickness(font_scale)

        text_width, text_height = 0, 0
        for text in text.splitlines():
            text_size = cv2.getTextSize(text=text,
                                        fontFace=font_face,
                                        fontScale=font_scale,
                                        thickness=font_thickness)[0]
            text_width = max(text_width, text_size[0])
            text_height += text_size[1]

        x, y = bbox_arr[0], bbox_arr[1]

        y_pos = position.value // 10
        if y_pos == 0:  # Y top
            y = bbox_arr[1] + text_height + self.TEXT_PADDING
        elif y_pos == 1:  # Y mid
            y = (bbox_arr[1] + bbox_arr[3]) // 2 + text_height // 2
        elif y_pos == 2:  # Y bottom
            y = bbox_arr[3] - text_height - self.TEXT_PADDING

        x_pos = position.value % 10
        if x_pos == 0:  # X Left
            x = bbox_arr[0] + self.TEXT_PADDING
        elif x_pos == 1:  # X mid
            x = (bbox_arr[0] + bbox_arr[2]) // 2 - text_width // 2
        elif x_pos == 2:  # X right
            x = bbox_arr[2] - self.TEXT_PADDING

        return x, y


    def _get_font_thickness(self, font_scale):
        return max(1, int(font_scale * 2))
    

    def _to_tuple(self, bbox: dai.ImgDetection, frame_shape: tuple) -> tuple[int, int, int, int]:
        tl, br = self._denormalize_bounding_box(bbox, frame_shape)
        return *tl, *br
    

    def _get_text_scale(self,
                       frame_shape: np.ndarray | tuple[int, ...], 
                       bbox: dai.ImgDetection | None = None
                       ) -> float:
        return min(1.0, min(frame_shape[:2]) / (1000 if bbox is None else 200))
    

    def generate_colors(self, number_of_colors: int, pastel=0.5) -> list[tuple[int, int, int]]:
        colors = []

        steps = math.ceil(math.sqrt(number_of_colors))

        for i in range(steps):
            hue = i / steps

            for j in range(steps):
                value = 0.6 + (j / steps) * 0.4

                r, g, b = colorsys.hsv_to_rgb(hue, pastel, value)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                colors.append((r, g, b))

        return colors[:number_of_colors]