#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
import blobconverter
from depthai_sdk.fps import FPSHandler

'''
Image quality assessment classifier demo running on device on RGB camera.
Classes: [Clean, Blur, Occlusion, Bright]

Run as:
python3 -m pip install -r requirements.txt
python3 main.py
'''

# --------------- Arguments ---------------

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn_path', type=str, help="select model blob path for inference, defaults to image_quality_assessment_256x256_001 from model zoo", default=None)

args = parser.parse_args()

NN_PATH = blobconverter.from_zoo(name="image_quality_assessment_256x256", zoo_type="depthai", shaves=6)
if args.nn_path: NN_PATH = args.nn_path

NN_WIDTH = 256
NN_HEIGHT = 256

N_CLASS_FRAMES_THRESHOLD = 10

# --------------- Utils ---------------

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
        
# --------------- Pipeline ---------------

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)
camRgb.setInterleaved(False)

# NN classifier
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(NN_PATH)
camRgb.preview.link(nn.input)

# Send class predictions from the NN to the host via XLink
nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

rgb_xout = pipeline.createXLinkOut()
rgb_xout.setStreamName("rgb")
camRgb.preview.link(rgb_xout.input)


with dai.Device(pipeline) as device:

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

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,0,255)
    thickness = 1
    
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # get predicted class
    def get_frame(imfFrame):
        '''
        Returns: detected_class_name, classes_prob
        '''
        detected_classes = np.array(imfFrame.getData()).view(np.float16)
        classes_prob = softmax(detected_classes)
        detected_class = np.argmax(classes_prob)

        return classes[detected_class], classes_prob

    fps = 0
    fps_handler = FPSHandler()

    while True:

        detected_class, classes_prob = get_frame(qNn.get())

        # update filter to smooth class changes
        class_filter.update(classes_prob=classes_prob)
        detected_class = class_filter.current_class

        rgb_frame = qCam.get().getCvFrame()

        fps_handler.tick("nn")
        fps = fps_handler.tickFps("nn")

        image = cv2.putText(rgb_frame, f"{detected_class} : FPS {fps:.2f}", (20, 20), font, fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.putText(image, " : ".join([class_name for class_name in classes.values()]), (20, 40), font, 0.3, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, " : ".join([f"{class_prob:.2f}" for class_prob in classes_prob]), (20, 55), font, 0.3, color, thickness, cv2.LINE_AA)

        cv2.imshow("Classification", image)

        if cv2.waitKey(1) == ord('q'):
            break
        