import argparse
from datetime import datetime, timedelta
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
    return np.array(nn_data.getFirstLayerFp16())


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


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
        self.detection_in = self.device.getInputQueue("detection_in")
        self.detection_nn = self.device.getOutputQueue("detection_nn")
        self.reid_in = self.device.getInputQueue("reid_in")
        self.reid_nn = self.device.getOutputQueue("reid_nn")
        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)

    def full_frame_cords(self, cords):
        original_cords = self.face_coords[0]
        return [
            original_cords[0 if i % 2 == 0 else 1] + val
            for i, val in enumerate(cords)
        ]

    def full_frame_bbox(self, bbox):
        relative_cords = self.full_frame_cords(bbox)
        height, width = self.frame.shape[:2]
        y_min = max(0, relative_cords[1])
        y_max = min(height, relative_cords[3])
        x_min = max(0, relative_cords[0])
        x_max = min(width, relative_cords[2])
        result_frame = self.frame[y_min:y_max, x_min:x_max]
        return result_frame, relative_cords

    def draw_bbox(self, bbox, color):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def run_detection(self):
        nn_data = run_nn(self.detection_in, self.detection_nn, {"input": to_planar(self.frame, (544, 320))})
        results = to_bbox_result(nn_data)
        self.pedestrian_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.4
        ]
        if len(self.pedestrian_coords) == 0:
            return []
        self.pedestrian_frames = [
            (coords, self.frame[coords[1]:coords[3], coords[0]:coords[2]])
            for coords in self.pedestrian_coords
        ]
        if debug:
            for bbox in self.pedestrian_coords:
                self.draw_bbox(bbox, (10, 245, 10))
        return self.pedestrian_frames

    def run_reid(self, detection_result):
        coords, detection = detection_result
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
            cv2.putText(self.debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
            if len(self.results_path[result_id]) > 1:
                cv2.polylines(self.debug_frame, [np.array(self.results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)

        return result_id

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        detection_results = self.run_detection()
        if len(detection_results) > 0:
            for detection_result in detection_results:
                person_id = self.run_reid(detection_result)

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
        del self.device


if __name__ == '__main__':
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
