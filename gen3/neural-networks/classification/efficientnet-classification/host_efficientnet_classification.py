import numpy as np
import depthai as dai
import cv2
from classes import class_names

LABELS = class_names()

class EfficientnetClassification(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output
              , debug: bool) -> "EfficientnetClassification":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        self.debug = debug
        return self

    # preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert (isinstance(preview, dai.ImgFrame))

        first_layer_name = nn.getAllLayerNames()[0]
        data = softmax(nn.getTensor(first_layer_name).astype(np.float16).flatten())
        result_conf = np.max(data)

        if result_conf > 0.2:
            result = {
                "name": LABELS[np.argmax(data)],
                "conf": round(100 * result_conf, 2)
            }
        else:
            result = None

        frame = preview.getCvFrame()
        color = (255, 0, 0)
        if result is not None:
            cv2.putText(frame, "{}".format(result["name"]), (5, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(frame, "Confidence: {}%".format(result["conf"]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX,0.4, color)

            if self.debug:
                print("{} ({}%)".format(result["name"], result["conf"]))

        cv2.imshow("Preview", cv2.resize(frame, (400, 400)))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
