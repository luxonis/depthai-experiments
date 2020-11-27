import argparse
import queue
import threading
import time
from datetime import datetime, timedelta
from multiprocessing import Pipe
from pathlib import Path
import cv2
import numpy as np
import depthai

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

debug = not args.no_debug

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is False and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")


def cos_dist(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    resized = cv2.resize(arr, shape)
    c1,c2,c3 = cv2.split(resized)
    return np.vstack(( c1,c2,c3 )).ravel()


def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())


def to_bbox_result(nn_data):
    arr = to_nn_result(nn_data)
    arr = arr[:np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr

def send_nn(x_in, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)


def run_nn(x_in, x_out, in_dict):
    send_nn(x_in, in_dict)
    has_results = wait_for_results(x_out)
    if not has_results:
        raise RuntimeError("No data from nn!")
    return x_out.get()


def frame_norm(frame, *xy_vals):
    height, width = frame.shape[:2]
    result = []
    for i, val in enumerate(xy_vals):
        if i % 2 == 0:
            result.append(max(0, min(width, int(val * width))))
        else:
            result.append(max(0, min(height, int(val * height))))
    return result


class ThreadedNode:
    EXIT_MESSAGE = "exit_message"

    def __init__(self, *args):
        # Just to limit number of messages in memory
        self.queue = queue.Queue(16)
        thread = threading.Thread(target=self.run_process, args=(self.queue, *args))
        thread.start()

    def run_process(self, queue, *args):
        self.queue = queue
        self.start(*args)
        self.run()

    def start(self, *args, **kwargs):
        pass

    def run(self):
        pass


class DetectionNode(ThreadedNode):
    def start(self, parent, device, reid_node):
        self.detection_nn = device.getOutputQueue("detection_nn")
        self.reid_in = device.getInputQueue("reid_in")
        self.parent = parent
        self.reid_node = reid_node
        self.pedestrian_coords = []

    def run(self):
        while True:
            data = self.queue.get()
            self.queue.task_done()
            results = to_bbox_result(self.detection_nn.get())
            self.pedestrian_coords = [
                frame_norm(data, *obj[3:7])
                for obj in results
                if obj[2] > 0.4
            ]

            pedestrian_frames = [
                (coords, data[coords[1]:coords[3], coords[0]:coords[2]])
                for coords in self.pedestrian_coords
            ]

            for coords, detection in pedestrian_frames:
                send_nn(self.reid_in, {"data": to_planar(detection, (48, 96))})
                self.reid_node.queue.put(coords)


class ReidentificationNode(ThreadedNode):
    def start(self, parent, device):
        self.reid_nn = device.getOutputQueue("reid_nn")
        self.results = {}
        self.results_path = {}
        self.results_time = {}
        self.parent = parent

    def run(self):
        while True:
            coords = self.queue.get()
            self.queue.task_done()
            nn_data = self.reid_nn.get()
            result = to_nn_result(nn_data)
            for key in list(self.results.keys()):
                if time.time() - self.results_time[key] > 4:
                    del self.results[key]
                    del self.results_time[key]
                    if debug:
                        del self.results_path[key]

            for person_id in self.results:
                dist = cos_dist(result, self.results[person_id])
                if dist > 0.5:
                    result_id = person_id
                    self.results[person_id] = result
                    break
            else:
                result_id = int(list(self.results)[-1]) + 1 if len(self.results) > 0 else 0
                self.results[result_id] = result
                if debug:
                    self.results_path[result_id] = []

            self.results_time[result_id] = time.time()

            if debug:
                x = (coords[0] + coords[2]) // 2
                y = (coords[1] + coords[3]) // 2
                self.results_path[result_id].append([x, y])


class Main:
    def __init__(self, file=None, camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            cam = self.pipeline.createColorCamera()
            cam.setPreviewSize(544, 320)
            cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setCamId(0)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)


        # NeuralNetwork
        print("Creating Person Detection Neural Network...")
        detection_in = self.pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_nn = self.pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013.blob").resolve().absolute()))
        detection_nn_xout = self.pipeline.createXLinkOut()
        detection_nn_xout.setStreamName("detection_nn")
        detection_in.out.link(detection_nn.input)
        detection_nn.out.link(detection_nn_xout.input)


        # NeuralNetwork
        print("Creating Person Reidentification Neural Network...")
        reid_in = self.pipeline.createXLinkIn()
        reid_in.setStreamName("reid_in")
        reid_nn = self.pipeline.createNeuralNetwork()
        reid_nn.setBlobPath(str(Path("models/person-reidentification-retail-0031.blob").resolve().absolute()))
        reid_nn_xout = self.pipeline.createXLinkOut()
        reid_nn_xout.setStreamName("reid_nn")
        reid_in.out.link(reid_nn.input)
        reid_nn.out.link(reid_nn_xout.input)

        print("Pipeline created.")

    def start_pipeline(self):
        self.device = depthai.Device()
        print("Starting pipeline...")
        self.device.startPipeline(self.pipeline)
        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)

        self.reid_node = ReidentificationNode(self, self.device)
        self.det_node = DetectionNode(self, self.device, self.reid_node)
        self.detection_in = self.device.getInputQueue("detection_in")

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        send_nn(self.detection_in, {"input": to_planar(self.frame, (544, 320))})
        self.det_node.queue.put(self.frame)

        # TODO - this has to be fixed. Data from other nodes isn't in sync with this stages iteration.
        # Suggestion, pass data all the way to the last node (or create last node)
        # and accumulate it all there - pass in all required things from previous stages,
        # where the data can then be visualized as below 
        if debug:
            for bbox in self.det_node.pedestrian_coords:
                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
            for result_id in self.reid_node.results:
                path = self.reid_node.results_path[result_id]
                if len(path) < 2:
                    continue
                x, y = path[-1]
                cv2.putText(self.debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                cv2.polylines(self.debug_frame, [np.array(path, dtype=np.int32)], False, (255, 0, 0), 2)

            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow("Camera_view", cv2.resize(self.debug_frame, ( int(900),  int(900 / aspect_ratio))))
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                raise StopIteration()

    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        tprev = time.time()
        frame_counter = 0
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break

            try:
                self.parse()
            except StopIteration:
                break

            ## Measures how quickly the frames are read from file.
            ## Because of the limit on Queue size, and the later
            ## stages being slower, the put function blocks and
            ## The rate caps at throughput of NN (detection) stage
             
            # frame_counter = frame_counter + 1
            # elapsed = time.time() - tprev
            # if elapsed >= 1.0 :
            #     tprev = time.time()
            #     print('FPS of first stage: ', frame_counter)
            #     frame_counter = 0

        cap.release()

    def run_camera(self):
        while True:
            self.frame = np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)
            try:
                self.parse()
            except StopIteration:
                break


    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device


if __name__ == '__main__':
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
