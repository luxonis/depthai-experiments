import cv2
import depthai as dai
import numpy as np
from fps import FPS

CAR_DETECTION_LABEL = 2

class CarDetection(dai.node.HostNode):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_THICKNESS = 2

    def __init__(self) -> None:
        self.fps_counter = FPS()
        super().__init__()


    def build(self, img_frame: dai.Node.Output, detections: dai.Node.Output) -> "CarDetection":
        self.link_args(img_frame, detections)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, img_frame, detections: dai.ImgDetections) -> None:
        assert(isinstance(img_frame, dai.ImgFrame))
        frame: np.ndarray = img_frame.getCvFrame()

        if detections is not None:
            for detection in detections.detections:
                if detection.label == CAR_DETECTION_LABEL:
                    tl, br = self.denormalize(detection.xmin, detection.ymin, detection.xmax, detection.ymax, frame.shape)
                    self.draw_bbox(frame, tl, br, (0, 255, 0), 1)
                
        text = f'FPS: {self.fps_counter.fps():.1f}'
        self.fps_counter.next_iter()

        x,y = self.get_text_relative_position(text, frame.shape)
        self.draw_text((x,y), text, frame)

        cv2.imshow("Car detection", frame)

        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()


    def denormalize(self, xmin, ymin, xmax, ymax, frame_shape) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Denormalize the bounding box to pixel coordinates (0..frame width, 0..frame height).
        Useful when you want to draw the bounding box on the frame.

        Args:
            frame_shape: Shape of the frame (height, width)

        Returns:
            Tuple of two points (top-left, bottom-right) in pixel coordinates
        """
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
                  alpha: float = 0.15
                  ) -> None:
        """
        Draw a rounded rectangle on the image (in-place).

        Args:
            img: Image to draw on.
            pt1: Top-left corner of the rectangle.
            pt2: Bottom-right corner of the rectangle.
            color: Rectangle color.
            thickness: Rectangle line thickness.
            r: Radius of the rounded corners.
            line_width: Width of the rectangle line.
            line_height: Height of the rectangle line.
        """
        x1, y1 = pt1
        x2, y2 = pt2

        line_width = np.abs(x2 - x1)

        line_height = np.abs(y2 - y1)
        
        # Top left
        cv2.line(img, (x1 , y1), (x1  + line_width, y1), color, thickness)
        cv2.line(img, (x1, y1 ), (x1, y1  + line_height), color, thickness)
        cv2.ellipse(img, (x1 , y1 ), (0, 0), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 , y1), (x2  - line_width, y1), color, thickness)
        cv2.line(img, (x2, y1 ), (x2, y1  + line_height), color, thickness)
        cv2.ellipse(img, (x2 , y1 ), (0, 0), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 , y2), (x1  + line_width, y2), color, thickness)
        cv2.line(img, (x1, y2 ), (x1, y2  - line_height), color, thickness)
        cv2.ellipse(img, (x1 , y2 ), (0, 0), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 , y2), (x2  - line_width, y2), color, thickness)
        cv2.line(img, (x2, y2 ), (x2, y2  - line_height), color, thickness)
        cv2.ellipse(img, (x2 , y2 ), (0, 0), 0, 0, 90, color, thickness)

        # Fill the area
        if alpha > 0:
            overlay = img.copy()

            thickness = -1
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])

            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[2], bbox[3])
            top_right = (bottom_right[0], top_left[1])
            bottom_left = (top_left[0], bottom_right[1])

            top_left_main_rect = (int(top_left[0] ), int(top_left[1]))
            bottom_right_main_rect = (int(bottom_right[0] ), int(bottom_right[1]))

            top_left_rect_left = (top_left[0], top_left[1] )
            bottom_right_rect_left = (bottom_left[0] , bottom_left[1] )

            top_left_rect_right = (top_right[0] , top_right[1] )
            bottom_right_rect_right = (bottom_right[0], bottom_right[1] )

            all_rects = [
                [top_left_main_rect, bottom_right_main_rect],
                [top_left_rect_left, bottom_right_rect_left],
                [top_left_rect_right, bottom_right_rect_right]
            ]

            [cv2.rectangle(overlay, pt1=rect[0], pt2=rect[1], color=color, thickness=thickness) for rect in all_rects]

            cv2.ellipse(overlay, (top_left[0] , top_left[1] ), (0, 0), 180.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (top_right[0] , top_right[1] ), (0, 0), 270.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (bottom_right[0] , bottom_right[1] ), (0, 0), 0.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (bottom_left[0] , bottom_left[1] ), (0, 0), 90.0, 0, 90, color, thickness)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


    def get_text_relative_position(self,
                              text: str,
                              frame_shape,
                              padding: int = 10
                              ) -> tuple[int, int]:
        """
        Get relative position of the text w.r.t. the bounding box.
        If bbox is None,the position is relative to the frame.
        """
        bbox = (0.0, 0.0, 1.0, 1.0)

        tl, br = self.denormalize(*bbox, frame_shape)

        bbox_arr = (*tl, *br)

        text_width, text_height = 0, 0
        for text in text.splitlines():
            text_size = cv2.getTextSize(text=text,
                                        fontFace=self.FONT,
                                        fontScale=self.FONT_SCALE,
                                        thickness=self.FONT_THICKNESS)[0]
            text_width = max(text_width, text_size[0])
            text_height += text_size[1]

        x, y = bbox_arr[0], bbox_arr[1]

        y = bbox_arr[1] + text_height + padding
        x = bbox_arr[0] + padding

        return x, y
    

    def draw_text(self, coords: tuple[int, int], text: str, frame: np.ndarray) -> None:
            shape = frame.shape[:2]

            font_scale = self.get_text_scale(shape)

            # Calculate font thickness
            font_thickness = max(1, int(font_scale * 2))

            dx, dy = cv2.getTextSize(text, self.FONT, font_scale, font_thickness)[0]
            dy += 10


            for line in text.splitlines():
                y = coords[1]

                # Background
                cv2.putText(img=frame,
                            text=line,
                            org=coords,
                            fontFace=self.FONT,
                            fontScale=font_scale,
                            color=(0, 0, 0),
                            thickness=font_thickness + 1,
                            lineType=cv2.LINE_AA)

                # Front text
                cv2.putText(img=frame,
                            text=line,
                            org=coords,
                            fontFace=self.FONT,
                            fontScale=font_scale,
                            color=(255, 255, 255),
                            thickness=font_thickness,
                            lineType=cv2.LINE_AA)

                coords = (coords[0], y + dy)


    def get_text_scale(self, frame_shape) -> float:
        return min(1.0, min(frame_shape) / 1000)