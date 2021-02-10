import argparse
import queue
import threading
from pathlib import Path

import cv2
import depthai
import numpy as np
from imutils.video import FPS
from math import cos, sin


parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

debug = not args.no_debug
camera = not args.video

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is False and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    if camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(544, 320)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Person Detection Neural Network...")
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013.blob").resolve().absolute()))
    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_nn.out.link(detection_nn_xout.input)
    if camera:
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath(
        str(Path("models/face-detection-retail-0004.blob").resolve().absolute()))
    face_in = pipeline.createXLinkIn()
    face_in.setStreamName("face_in")
    face_in.out.link(face_nn.input)
    face_nn_xout = pipeline.createXLinkOut()
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)

    # NeuralNetwork
    print("Creating Landmarks Detection Neural Network...")
    land_nn = pipeline.createNeuralNetwork()
    land_nn.setBlobPath(
        str(Path("models/landmarks-regression-retail-0009.blob").resolve().absolute())
    )
    land_nn_xin = pipeline.createXLinkIn()
    land_nn_xin.setStreamName("landmark_in")
    land_nn_xin.out.link(land_nn.input)
    land_nn_xout = pipeline.createXLinkOut()
    land_nn_xout.setStreamName("landmark_nn")
    land_nn.out.link(land_nn_xout.input)

    print("Pipeline created.")
    return pipeline


class Main:
    def __init__(self, device):
        self.device = device
        print("Starting pipeline...")
        self.device.startPipeline()
        if camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)
        else:
            self.cap = cv2.VideoCapture(str(Path("./input.mp4").resolve().absolute()))
            self.detection_in = self.device.getInputQueue("detection_in", maxSize=1, blocking=False)
        self.det_box_q = queue.Queue()
        self.face_box_q = queue.Queue()
        self.frame = None
        self.det_bboxes = []
        self.face_bboxes = []
        self.land_data = None
        self.test = None
        self.running = True
        self.fps = FPS()
        self.fps.start()

    def det_thread(self):
        detection_nn = self.device.getOutputQueue("detection_nn", maxSize=1, blocking=False)
        face_in = self.device.getInputQueue("face_in", maxSize=1, blocking=False)

        while self.running:
            if self.frame is None:
                continue
            try:
                det_bboxes = np.array(detection_nn.get().getFirstLayerFp16())
            except RuntimeError:
                continue
            det_bboxes = det_bboxes.reshape((det_bboxes.size // 7, 7))
            det_bboxes = det_bboxes[det_bboxes[:, 2] > 0.7][:, 3:7]
            if len(det_bboxes) == 0:
                continue
            self.det_bboxes = np.apply_along_axis(lambda raw_bbox: frame_norm(self.frame, raw_bbox), 1, det_bboxes)

            for bbox in self.det_bboxes:
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                nn_data = depthai.NNData()
                nn_data.setLayer("input", to_planar(det_frame, (300, 300)))
                face_in.send(nn_data)
                self.det_box_q.put(bbox)

    def face_thread(self):
        face_nn = self.device.getOutputQueue("face_nn", maxSize=1, blocking=False)
        landmark_in = self.device.getInputQueue("landmark_in", maxSize=1, blocking=False)

        while self.running:
            if self.frame is None:
                continue
            try:
                face_bboxes = np.array(face_nn.get().getFirstLayerFp16())
            except RuntimeError:
                continue
            face_bboxes = face_bboxes.reshape((face_bboxes.size // 7, 7))
            face_bboxes = face_bboxes[face_bboxes[:, 2] > 0.7][:, 3:7]
            try:
                det_bbox = self.det_box_q.get(block=True, timeout=100)
            except queue.Empty:
                continue
            self.det_box_q.task_done()
            det_frame = self.frame[det_bbox[1]:det_bbox[3], det_bbox[0]:det_bbox[2]]
            if len(face_bboxes) == 0:
                continue
            face_bboxes = np.apply_along_axis(lambda raw_bbox: frame_norm(det_frame, raw_bbox), 1, face_bboxes)
            left = det_bbox[0]
            top = det_bbox[1]
            face_bboxes[:, ::2] += left
            face_bboxes[:, 1::2] += top
            self.face_bboxes = face_bboxes

            for bbox in self.face_bboxes:
                face_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                nn_data = depthai.NNData()
                nn_data.setLayer("0", to_planar(face_frame, (48, 48)))
                landmark_in.send(nn_data)
                self.face_box_q.put(bbox)

    def land_thread(self):
        land_nn = self.device.getOutputQueue("landmark_nn", maxSize=1, blocking=False)

        while self.running:
            try:
                land_in = land_nn.get().getFirstLayerFp16()
            except RuntimeError:
                continue

            try:
                face_bbox = self.face_box_q.get(block=True, timeout=100)
            except queue.Empty:
                continue

            self.face_box_q.task_done()
            left = face_bbox[0]
            top = face_bbox[1]
            face_frame = self.frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            land_data = frame_norm(face_frame, land_in)
            land_data[::2] += left
            land_data[1::2] += top
            self.land_data = land_data

    def should_run(self):
        return True if camera else self.cap.isOpened()

    def get_frame(self, retries=0):
        if camera:
            return True, np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)
        else:
            read_correctly, new_frame = self.cap.read()
            if not read_correctly or new_frame is None:
                if retries < 5:
                    return self.get_frame(retries+1)
                else:
                    print("Source closed, terminating...")
                    return False, None
            else:
                return read_correctly, new_frame

    def run(self):
        self.threads = [
            threading.Thread(target=self.det_thread),
            threading.Thread(target=self.face_thread),
            threading.Thread(target=self.land_thread)
        ]
        for thread in self.threads:
            thread.start()

        while self.should_run():
            read_correctly, new_frame = self.get_frame()
            
            if not read_correctly:
                break

            self.fps.update()
            self.frame = new_frame
            self.debug_frame = self.frame.copy()

            if not camera:
                nn_data = depthai.NNData()
                nn_data.setLayer("input", to_planar(self.frame, (544, 320)))
                self.detection_in.send(nn_data)

            if debug:
                for det_bbox in self.det_bboxes:   # people
                    cv2.rectangle(self.debug_frame, (det_bbox[0], det_bbox[1]), (det_bbox[2], det_bbox[3]), (10, 245, 10), 2)
                for face_bbox in self.face_bboxes:   # faces
                    cv2.rectangle(self.debug_frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (10, 245, 10), 2)
                if self.land_data is not None:
                    cv2.circle(self.debug_frame, tuple(self.land_data[:2]), 1, (255, 255, 0))  # Right eye
                    cv2.circle(self.debug_frame, tuple(self.land_data[2:4]), 1, (255, 255, 0))  # Left eye
                    cv2.circle(self.debug_frame, tuple(self.land_data[4:6]), 1, (255, 255, 0))  # Nose
                    cv2.circle(self.debug_frame, tuple(self.land_data[6:8]), 1, (255, 255, 0))  # Right mouth
                    cv2.circle(self.debug_frame, tuple(self.land_data[8:]), 1, (255, 255, 0))  # Left mouth

                if camera:
                    cv2.imshow("Camera view", self.debug_frame)
                else:
                    aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
                    cv2.imshow("Video view", cv2.resize(self.debug_frame, (int(900),  int(900 / aspect_ratio))))
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

        self.fps.stop()
        print("FPS: {:.2f}".format(self.fps.fps()))
        if not camera:
            self.cap.release()
        cv2.destroyAllWindows()
        for i in range(1, 5):  # https://stackoverflow.com/a/25794701/5494277
            cv2.waitKey(1)
        self.running = False


with depthai.Device(create_pipeline()) as device:
    app = Main(device)
    app.run()

for thread in app.threads:
    thread.join()
