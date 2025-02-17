import threading

import cv2
import depthai as dai
import numpy as np
from utils.server import ThreadedHTTPServer, VideoStreamHandler

HTTP_SERVER_PORT = 8083


FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 255, 0)


class MJPEGStreamer(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self, preview: dai.Node.Output, nn: dai.Node.Output, labels: list[str]
    ) -> "MJPEGStreamer":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        # Start server
        self.server = ThreadedHTTPServer(
            ("localhost", HTTP_SERVER_PORT), VideoStreamHandler
        )
        self.labels = labels
        th = threading.Thread(target=self.server.serve_forever)
        th.daemon = True
        th.start()
        print("To view the MJPEG stream go to http://localhost:8083")

        return self

    def _draw_detection(self, img: np.ndarray, detection: dai.ImgDetection):
        height, width, _ = img.shape

        # Denormalize bounding box
        x1 = int(detection.xmin * width)
        x2 = int(detection.xmax * width)
        y1 = int(detection.ymin * height)
        y2 = int(detection.ymax * height)

        try:
            label = self.labels[detection.label]
        except Exception as _:
            label = detection.label

        new_img = cv2.putText(img, str(label), (x1 + 10, y1 + 15), FONT, 0.5, COLOR, 1)
        new_img = cv2.putText(
            new_img,
            str(round(detection.confidence * 100, 2)),
            (x1 + 10, y1 + 35),
            FONT,
            0.5,
            COLOR,
            1,
        )
        new_img = cv2.rectangle(new_img, (x1, y1), (x2, y2), COLOR, FONT)
        return new_img

    def process(self, preview: dai.Buffer, nn: dai.ImgDetections) -> None:
        assert isinstance(preview, dai.ImgFrame)

        frame = preview.getCvFrame().copy()

        for detection in nn.detections:
            frame = self._draw_detection(frame, detection)

        self.server.frametosend = frame
