import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera

NN_WIDTH = 256
NN_HEIGHT = 256

N_CLASS_FRAMES_THRESHOLD = 10


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ClassSmoothFilter:
    def __init__(self, classes, n_frames_threshold=10):
        self.classes = classes
        self.n_frames_threshold = n_frames_threshold
        self.classes_frame_counter = {
            class_: 0 for class_ in classes.values()
        }
        self.current_class = None

    def init_classes_frame_counter(self):
        self.classes_frame_counter = {
            class_: 0 for class_ in classes.values()
        }

    def update(self, classes_prob):
        top_class_idx = np.argmax(classes_prob)
        top_class = self.classes[top_class_idx]

        self.classes_frame_counter[top_class] += 1

        if self.classes_frame_counter[top_class] >= self.n_frames_threshold:
            self.current_class = top_class
            self.init_classes_frame_counter()


classes = {
    0: "Clean",
    1: "Blur",
    2: "Occlusion",
    3: "Bright"
}
class_filter = ClassSmoothFilter(
    classes=classes,
    n_frames_threshold=N_CLASS_FRAMES_THRESHOLD
)


def callback(packet):
    frame = packet.frame
    nn_data = packet.img_detections

    detected_classes = np.array(nn_data.getData()).view(np.float16)
    classes_prob = softmax(detected_classes)
    detected_class = np.argmax(classes_prob)

    class_filter.update(classes_prob=classes_prob)
    label = class_filter.current_class

    cv2.putText(frame, label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(frame, " : ".join([class_name for class_name in classes.values()]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, " : ".join([f"{class_prob:.2f}" for class_prob in classes_prob]),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Image Quality Assessment', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    nn_path = blobconverter.from_zoo(name='image_quality_assessment_256x256', shaves=6, zoo_type='depthai')
    nn = oak.create_nn(nn_path, color)
    nn.config_nn(resize_mode='crop')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
