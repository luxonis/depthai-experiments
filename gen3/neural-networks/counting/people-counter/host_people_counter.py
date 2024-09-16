import cv2
import depthai as dai
import numpy as np


class PeopleCounter(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "PeopleCounter":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()

        detections = nn.getTensor("detection_out").astype(np.float16).reshape(-1, 7)
        people = 0

        for detection in detections:
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if confidence > 0.6:
                people += 1
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color = (0, 0, 255), thickness=2)
                cv2.putText(frame, f"{confidence*100:.2f}%", (xmin + 15, ymin + 25)
                            , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2)

        cv2.putText(frame, f"People count: {people}", (20, 60)
                    , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=2)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
