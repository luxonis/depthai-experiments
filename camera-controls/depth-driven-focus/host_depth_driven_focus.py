import numpy as np
import cv2
import depthai as dai
import math

DEBUG = True
LENS_STEP = 3


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords) -> None:
        cv2.putText(
            frame, text, coords, self.text_type, 1.5, self.bg_color, 6, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 1.5, self.color, 2, self.line_type
        )

    def rectangle(self, frame, x1, y1, x2, y2) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 6)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)


class DepthDrivenFocus(dai.node.HostNode):
    def __init__(self) -> None:
        self._lensPos = 150
        self._lensMin = 0
        self._lensMax = 255
        self._video_width = 1080
        self._video_height = 1080
        self._text = TextHelper()
        super().__init__()

    def build(
        self,
        preview: dai.Node.Output,
        control_queue: dai.Node.Output,
        face_detection: dai.Node.Output,
        face_detection_depth: dai.Node.Output,
    ) -> "DepthDrivenFocus":
        self.link_args(preview, face_detection, face_detection_depth)
        self.control_queue = control_queue
        self.sendProcessingToPipeline(True)
        return self

    def process(
        self,
        message: dai.ImgFrame,
        face_detection: dai.SpatialImgDetections,
        detection_depth: dai.ImgFrame,
    ) -> None:
        frame: np.ndarray = message.getCvFrame()

        if DEBUG:
            depthFrame = detection_depth.getFrame()
            depthFrame = cv2.pyrDown(depthFrame)
            depthFrame = cv2.normalize(
                depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
            )
            depthFrame = cv2.equalizeHist(depthFrame)
            depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_HOT)

        closest_dist = 99999999
        for detection in face_detection.detections:
            x1 = int(detection.xmin * self._video_width)
            x2 = int(detection.xmax * self._video_width)
            y1 = int(detection.ymin * self._video_height)
            y2 = int(detection.ymax * self._video_height)

            dist = int(calculate_distance(detection.spatialCoordinates))
            if dist < closest_dist:
                closest_dist = dist
            self._text.rectangle(frame, x1, y1, x2, y2)

            if DEBUG:
                roi = detection.boundingBoxMapping.roi
                roi = roi.denormalize(depthFrame.shape[1], depthFrame.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)
                self._text.rectangle(depthFrame, xmin, ymin, xmax, ymax)

        if closest_dist != 99999999:
            self._text.putText(
                frame,
                "Face distance: {:.2f} m".format(closest_dist / 1000),
                (330, 1045),
            )
            new_lens_pos = max(
                self._lensMin, min(get_lens_position(closest_dist), self._lensMax)
            )
            if new_lens_pos != self._lensPos and new_lens_pos != 255:
                self._lensPos = new_lens_pos

                print("Setting automatic focus, lens position: ", new_lens_pos)
                ctrl = dai.CameraControl()
                ctrl.setManualFocus(self._lensPos)
                self.control_queue.send(ctrl)
            self._text.putText(frame, f"Lens position: {self._lensPos}", (330, 1000))
        else:
            self._text.putText(frame, "Face distance: /", (330, 1045))
            self._text.putText(frame, f"Lens position: {self._lensPos}", (330, 1000))

        cv2.imshow("Preview", cv2.resize(frame, (750, 750)))

        if DEBUG:
            cv2.imshow("Depth", depthFrame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()

        elif key in [ord(","), ord(".")]:
            if key == ord(","):
                self._lensPos -= LENS_STEP
            if key == ord("."):
                self._lensPos += LENS_STEP
            self._lensPos = max(self._lensMin, min(self._lensPos, self._lensMax))

            print("Setting manual focus, lens position: ", self._lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(self._lensPos)
            self.control_queue.send(ctrl)


def get_lens_position(dist):
    # = 150 - A10*0.0242 + 0.00000412*A10^2
    return int(150 - dist * 0.0242 + 0.00000412 * dist**2)


def calculate_distance(coords):
    return math.sqrt(coords.x**2 + coords.y**2 + coords.z**2)
