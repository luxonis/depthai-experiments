import argparse
from datetime import datetime, timedelta
from multiprocessing import Pipe, Process
from pathlib import Path
import cv2
import numpy as np
from math import cos, sin
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
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_nn_result(nn_data):
    print(nn_data, nn_data.getAllLayerNames(), dir(nn_data))
    return np.array(nn_data.getFirstLayerFp16())


def to_bbox_result(nn_data):
    arr = to_nn_result(nn_data)
    arr = arr[:np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
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
        self.mainPipe, self.subPipe = Pipe()
        self.p = Process(target=self.run_process, args=(self.subPipe, *args))
        self.p.start()

    def run_process(self, pipe, *args):
        self.start(*args)
        while True:
            val = pipe.recv()
            if isinstance(val, str) and val == self.EXIT_MESSAGE:
                return
            self.accept(val)

    def start(self, *args, **kwargs):
        pass

    def accept(self, data):
        pass

    def receive(self, data):
        self.mainPipe.send(data)

    def exit(self):
        self.mainPipe.send(self.EXIT_MESSAGE)


class DetectionNode(ThreadedNode):
    def start(self, parent, device, reid_node):
        self.detection_in = device.getInputQueue("detection_in")
        self.detection_nn = device.getOutputQueue("detection_nn")
        self.parent = parent
        self.reid_node = reid_node

    def accept(self, data):
        nn_data = run_nn(self.detection_in, self.detection_nn, {"input": to_planar(data, (544, 320))})
        results = to_bbox_result(nn_data)
        pedestrian_coords = [
            frame_norm(data, *obj[3:7])
            for obj in results
            if obj[2] > 0.4
        ]

        pedestrian_frames = [
            (coords, data[coords[1]:coords[3], coords[0]:coords[2]])
            for coords in pedestrian_coords
        ]
        if debug:
            for bbox in pedestrian_coords:
                cv2.rectangle(self.parent.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

        self.reid_node.receive(pedestrian_frames)


class ReidentificationNode(ThreadedNode):
    def start(self, parent, device):
        self.reid_in = device.getInputQueue("reid_in")
        self.reid_nn = device.getOutputQueue("reid_nn")
        self.results = {}
        self.results_path = {}
        self.parent = parent

    def accept(self, data):
        coords, detection = data
        nn_data = run_nn(self.reid_in, self.reid_nn, {"data": to_planar(detection, (48, 96))})
        result = to_nn_result(nn_data)
        for person_id in self.results:
            dist = cos_dist(result, self.results[person_id])
            if dist > 0.5:
                result_id = person_id
                self.results[person_id] = result
                break
        else:
            result_id = len(self.results)
            self.results[result_id] = result
            if debug:
                self.results_path[result_id] = []

        if debug:
            x = (coords[0] + coords[2]) // 2
            y = (coords[1] + coords[3]) // 2
            self.results_path[result_id].append([x, y])
            cv2.putText(self.parent.debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
            if len(self.results_path[result_id]) > 1:
                cv2.polylines(self.parent.debug_frame, [np.array(self.results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)


class Main:
    def __init__(self, file=None, camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.results = {}
        self.results_path = {}
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

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        self.det_node.receive(self.frame)

        if debug:
            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow("Camera_view", cv2.resize(self.debug_frame, ( int(900),  int(900 / aspect_ratio))))
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                raise StopIteration()

    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break

            try:
                self.parse()
            except StopIteration:
                break

        cap.release()

    def run_camera(self):
        while True:
            self.frame = np.array(self.cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
            try:
                self.parse()
            except StopIteration:
                break


    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        self.det_node.exit()
        self.reid_node.exit()
        del self.device


if __name__ == '__main__':
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
