import numpy as np
import depthai as dai
import cv2

# MobilenetSSD label texts
LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 0, 255)


class StreamOutput(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        preview: dai.Node.Output,
        depth: dai.Node.Output,
        nn: dai.Node.Output,
        server,
    ) -> "StreamOutput":
        self.link_args(preview, depth, nn)
        self.sendProcessingToPipeline(True)

        self.server = server
        return self

    def process(
        self, preview: dai.ImgFrame, depth: dai.ImgFrame, nn: dai.SpatialImgDetections
    ) -> None:
        frame = preview.getCvFrame()
        depth_frame = depth.getCvFrame()

        depth_frame = cv2.normalize(
            depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
        )
        depth_frame = cv2.equalizeHist(depth_frame)
        depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)

        height = frame.shape[0]
        width = frame.shape[1]

        for detection in nn.detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = LABELS[detection.label]
            except Exception as _:
                label = detection.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), FONT, 0.5, COLOR)
            cv2.putText(
                frame,
                "{:.2f}".format(detection.confidence * 100),
                (x1 + 10, y1 + 35),
                FONT,
                0.5,
                COLOR,
            )
            cv2.putText(
                frame,
                f"X: {int(detection.spatialCoordinates.x)} mm",
                (x1 + 10, y1 + 50),
                FONT,
                0.5,
                COLOR,
            )
            cv2.putText(
                frame,
                f"Y: {int(detection.spatialCoordinates.y)} mm",
                (x1 + 10, y1 + 65),
                FONT,
                0.5,
                COLOR,
            )
            cv2.putText(
                frame,
                f"Z: {int(detection.spatialCoordinates.z)} mm",
                (x1 + 10, y1 + 80),
                FONT,
                0.5,
                COLOR,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, FONT)

        new_width = int(depth_frame.shape[1] * (frame.shape[0] / depth_frame.shape[0]))
        stacked = np.hstack(
            [frame, cv2.resize(depth_frame, (new_width, frame.shape[0]))]
        )
        cv2.imshow("Preview", stacked)
        self.server.frametosend = stacked

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()
