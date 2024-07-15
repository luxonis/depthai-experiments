import cv2
import depthai as dai
import numpy as np

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

class ImageClassification(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output
              , debug: bool) -> "ImageClassification":
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

        if result_conf > 0.5:
            result = {
                "name": CLASSES[np.argmax(data)],
                "conf": round(100 * result_conf, 2)
            }
        else:
            result = None

        frame = preview.getCvFrame()

        color = (255, 0, 0)
        if result is not None:
            cv2.putText(frame, "{}".format(result["name"]), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "Confidence: {}%".format(result["conf"]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            if self.debug:
                print("{} ({}%)".format(result["name"], result["conf"]))

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
