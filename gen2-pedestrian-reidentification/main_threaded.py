import time
import queue
import threading
from pathlib import Path

import cv2
import depthai
import numpy as np
from imutils.video import FPS


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    tstart = time.monotonic()
    planar = [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]
    tdiff = time.monotonic() - tstart
    #print('time take to planar: ', tdiff)
    return planar

def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    # NeuralNetwork
    print("Creating Person Detection Neural Network...")
    detection_in = pipeline.createXLinkIn()
    detection_in.setStreamName("detection_in")
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013.blob").resolve().absolute()))
    # Increase threads for detection
    detection_nn.setNumInferenceThreads(2)

    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_in.out.link(detection_nn.input)
    detection_nn.out.link(detection_nn_xout.input)

    # NeuralNetwork
    print("Creating Person Reidentification Neural Network...")
    reid_in = pipeline.createXLinkIn()
    reid_in.setStreamName("reid_in")
    reid_nn = pipeline.createNeuralNetwork()
    reid_nn.setBlobPath(str(Path("models/person-reidentification-retail-0031.blob").resolve().absolute()))
    
    # Decrease threads for reidentification
    reid_nn.setNumInferenceThreads(1)
    
    reid_nn_xout = pipeline.createXLinkOut()
    reid_nn_xout.setStreamName("reid_nn")
    reid_in.out.link(reid_nn.input)
    reid_nn.out.link(reid_nn_xout.input)

    print("Pipeline created.")
    return pipeline


class Main:
    def __init__(self):
        self.device = depthai.Device(create_pipeline())
        print("Starting pipeline...")
        self.device.startPipeline()
        self.detection_in = self.device.getInputQueue("detection_in", 4, False)
        self.reid_in = self.device.getInputQueue("reid_in")

        self.bboxes = []
        self.results = {}
        self.results_path = {}
        self.reid_bbox_q = queue.Queue()
        self.next_id = 0

        self.cap = cv2.VideoCapture(str(Path("./input.mp4").resolve().absolute()))

        self. = queue.Queue()


        self.fps = FPS()
        self.fps.start()

    def det_thread(self):
        detection_nn = self.device.getOutputQueue("detection_nn")
        while True:
            bboxes = np.array(detection_nn.get().getFirstLayerFp16())
            bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

            # Let reid stage know how many boxes were found
            reid_bbox_num_q.put(len(bboxes))

            for raw_bbox in bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                nn_data = depthai.NNData()
                nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                self.reid_in.send(nn_data)
                self.reid_bbox_q.put(bbox)

    def reid_thread(self):
        reid_nn = self.device.getOutputQueue("reid_nn")
        while True:
            reid_result = reid_nn.get().getFirstLayerFp16()
            bbox = self.reid_bbox_q.get()

            for person_id in self.results:
                dist = cos_dist(reid_result, self.results[person_id])
                if dist > 0.5:
                    result_id = person_id
                    self.results[person_id] = reid_result
                    break
            else:
                result_id = self.next_id
                self.results[result_id] = reid_result
                self.results_path[result_id] = []
                self.next_id += 1

            

            cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
            x = (bbox[0] + bbox[2]) // 2
            y = (bbox[1] + bbox[3]) // 2
            self.results_path[result_id].append([x, y])
            cv2.putText(self.debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
            if len(self.results_path[result_id]) > 1:
                cv2.polylines(self.debug_frame, [np.array(self.results_path[result_id], dtype=np.int32)], False,
                              (255, 0, 0), 2)

    def visualization_thread(self):
        
        # Wait to receive first inference frame (baseline latency)
        inferenceFrame = self.vis_inference_frame_q.get()

        while True:
            # Wait to receive frame
            frame = self.vis_inference_frame_q.get()

            # Try to receive infered frame
            recInferenceFrame = self.vis_inference_frame_q.tryGet()
            if recInferenceFrame != None:
                inferenceFrame = recInferenceFrame
            
            # inferenceFrame containes latest 



            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow("Camera_view", cv2.resize(self.debug_frame, (int(900),  int(900 / aspect_ratio))))





    def run(self):
        threading.Thread(target=self.det_thread, daemon=True).start()
        threading.Thread(target=self.reid_thread, daemon=True).start()

        t1 = time.monotonic()
        while self.cap.isOpened():
            read_correctly, self.frame = self.cap.read()

            if not read_correctly:
                break

            self.fps.update()
            self.debug_frame = self.frame.copy()

            nn_data = depthai.NNData()
            nn_data.setLayer("input", to_planar(self.frame, (544, 320)))
            self.detection_in.send(nn_data)


            # 30 FPS
            if cv2.waitKey(30) == ord('q'):
                cv2.destroyAllWindows()
                break

            s1diff = time.monotonic() - t1
            t1 = time.monotonic()
            print('stage 1 ms: ', s1diff * 1000)

        self.fps.stop()
        print("FPS: {:.2f}".format(self.fps.fps()))
        self.cap.release()


Main().run()
