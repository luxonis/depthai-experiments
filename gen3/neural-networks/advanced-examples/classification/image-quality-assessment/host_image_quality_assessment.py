import numpy as np
import depthai as dai
import cv2

CLASS_FRAMES_THRESHOLD = 10

CLASSES = {
    0: "Clean",
    1: "Blur",
    2: "Occlusion",
    3: "Bright"
}

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255,0,255)
thickness = 1

class ClassSmoothFilter:
    def __init__(self):
        self.classes_frame_counter = { class_: 0 for class_ in CLASSES.values() }
        self.current_class = None

    def init_classes_frame_counter(self):
        self.classes_frame_counter = { class_: 0 for class_ in CLASSES.values() }

    def update(self, classes_prob):
        top_class_idx = np.argmax(classes_prob)
        top_class = CLASSES[top_class_idx]

        self.classes_frame_counter[top_class] += 1

        if self.classes_frame_counter[top_class] >= CLASS_FRAMES_THRESHOLD:
            self.current_class = top_class
            self.init_classes_frame_counter()


class ImageQualityAssessment(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.class_filter = ClassSmoothFilter()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "ImageQualityAssessment":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        detected_classes = np.array(nn.getData()).view(np.float16)
        classes_prob = softmax(detected_classes)

        self.class_filter.update(classes_prob=classes_prob)
        detected_class = self.class_filter.current_class

        frame = preview.getCvFrame()
        cv2.putText(frame, f"{detected_class} ", (20, 20), font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.putText(frame, " : ".join([class_name for class_name in CLASSES.values()]), (20, 40), font, 0.3,
                            color, thickness, cv2.LINE_AA)
        cv2.putText(frame, " : ".join([f"{class_prob:.2f}" for class_prob in classes_prob]), (20, 55), font,
                            0.3, color, thickness, cv2.LINE_AA)

        cv2.imshow("Classification", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
