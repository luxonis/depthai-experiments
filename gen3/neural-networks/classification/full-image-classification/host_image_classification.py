import numpy as np
import depthai as dai
import cv2
import efficientnet_classes


class ImageClassification(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, efficientnet: bool) -> "ImageClassification":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        self.efficientnet = efficientnet
        if efficientnet:
            self.classes = efficientnet_classes.class_names()
        else:
            self.classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        return self

    # preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert (isinstance(preview, dai.ImgFrame))

        frame = preview.getCvFrame()
        data = softmax(nn.getFirstTensor().flatten())

        result_conf = np.max(data)
        if result_conf > (0.2 if self.efficientnet else 0.5):
            result = {
                "name": self.classes[np.argmax(data)],
                "conf": round(100 * result_conf, 2)
            }
        else:
            result = None

        if result is not None:
            cv2.putText(frame, "{}".format(result["name"])
                        , (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
            cv2.putText(frame, "Confidence: {:.2f}%".format(result["conf"])
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255))

        # For higher quality output, send
        cv2.imshow("Preview", cv2.resize(frame, (400, 400)))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
