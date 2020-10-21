import argparse
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
from math import cos, sin
import depthai

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', type=int, help="Camera ID to be used for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

debug = not args.no_debug

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is None and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam <cam_id>\" to run on DepthAI camera or \"-vid <path>\" to run on video file")


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


def draw_3d_axis(image, head_pose, origin, size=50):
    roll = head_pose[0] * np.pi / 180
    pitch = head_pose[1] * np.pi / 180
    yaw = -(head_pose[2] * np.pi / 180)

    # X axis (red)
    x1 = size * (cos(yaw) * cos(roll)) + origin[0]
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = size * (-cos(yaw) * sin(roll)) + origin[0]
    y2 = size * (-cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = size * (-sin(yaw)) + origin[0]
    y3 = size * (cos(yaw) * sin(pitch)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x3), int(y3)), (255, 0, 0), 2)

    return image


class Main:
    def __init__(self, file=None, camera=None):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        if self.camera is not None:
            # ColorCamera
            print("Creating Color Camera...")
            cam = self.pipeline.createColorCamera()
            cam.setPreviewSize(300, 300)
            cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setCamId(self.camera)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)


        # NeuralNetwork
        print("Creating Face Detection Neural Network...")
        face_in = self.pipeline.createXLinkIn()
        face_in.setStreamName("face_in")
        face_nn = self.pipeline.createNeuralNetwork()
        face_nn.setBlobPath(str(Path("models/face-detection-retail-0004/face-detection-retail-0004.blob").resolve().absolute()))
        face_nn_xout = self.pipeline.createXLinkOut()
        face_nn_xout.setStreamName("face_nn")
        face_in.out.link(face_nn.input)
        face_nn.out.link(face_nn_xout.input)
        
        # NeuralNetwork
        print("Creating Landmarks Detection Neural Network...")
        land_nn = self.pipeline.createNeuralNetwork()
        land_nn.setBlobPath(
            str(Path("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob").resolve().absolute())
        )
        land_nn_xin = self.pipeline.createXLinkIn()
        land_nn_xin.setStreamName("landmark_in")
        land_nn_xin.out.link(land_nn.input)
        land_nn_xout = self.pipeline.createXLinkOut()
        land_nn_xout.setStreamName("landmark_nn")
        land_nn.out.link(land_nn_xout.input)

        # NeuralNetwork
        print("Creating Head Pose Neural Network...")
        pose_nn = self.pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(
            str(Path("models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.blob").resolve().absolute())
        )
        pose_nn_xin = self.pipeline.createXLinkIn()
        pose_nn_xin.setStreamName("pose_in")
        pose_nn_xin.out.link(pose_nn.input)
        pose_nn_xout = self.pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        # NeuralNetwork
        print("Creating Gaze Estimation Neural Network...")
        gaze_nn = self.pipeline.createNeuralNetwork()
        gaze_nn.setBlobPath(
            str(Path("models/gaze-estimation-adas-0002/gaze-estimation-adas-0002.blob").resolve().absolute())
        )
        gaze_nn_xin = self.pipeline.createXLinkIn()
        gaze_nn_xin.setStreamName("gaze_in")
        gaze_nn_xin.out.link(gaze_nn.input)
        gaze_nn_xout = self.pipeline.createXLinkOut()
        gaze_nn_xout.setStreamName("gaze_nn")
        gaze_nn.out.link(gaze_nn_xout.input)

        print("Pipeline created.")

    def start_pipeline(self):
        self.device = depthai.Device()
        print("Starting pipeline...")
        self.device.startPipeline(self.pipeline)
        self.face_in = self.device.getInputQueue("face_in")
        self.face_nn = self.device.getOutputQueue("face_nn")
        self.land_in = self.device.getInputQueue("landmark_in")
        self.land_nn = self.device.getOutputQueue("landmark_nn")
        self.pose_in = self.device.getInputQueue("pose_in")
        self.pose_nn = self.device.getOutputQueue("pose_nn")
        self.gaze_in = self.device.getInputQueue("gaze_in")
        self.gaze_nn = self.device.getOutputQueue("gaze_nn")
        if self.camera is not None:
            self.cam_out = self.device.getOutputQueue("cam_out")

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

    def run_face(self):
        nn_data = run_nn(self.face_in, self.face_nn, {"data": to_planar(self.frame, (300, 300))})
        results = to_bbox_result(nn_data)
        self.face_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.4
        ]
        if len(self.face_coords) == 0:
            return False
        self.face_frame = self.frame[
            self.face_coords[0][1]:self.face_coords[0][3],
            self.face_coords[0][0]:self.face_coords[0][2]
        ]
        if debug:
            for bbox in self.face_coords:
                self.draw_bbox(bbox, (10, 245, 10))
        return True

    def run_landmark(self):
        nn_data = run_nn(self.land_in, self.land_nn, {"0": to_planar(self.face_frame, (48, 48))})
        out = frame_norm(self.face_frame, *to_nn_result(nn_data))
        raw_left, raw_right, raw_nose = out[:2], out[2:4], out[4:6]

        self.left_eye_image, self.left_eye_bbox = self.full_frame_bbox([
            raw_left[0] - 30, raw_left[1] - 30, raw_left[0] + 30, raw_left[1] + 30
        ])
        self.right_eye_image, self.right_eye_bbox = self.full_frame_bbox([
            raw_right[0] - 30, raw_right[1] - 30, raw_right[0] + 30, raw_right[1] + 30
        ])
        self.nose = self.full_frame_cords(raw_nose)

        if debug:
            cv2.circle(self.debug_frame, (self.nose[0], self.nose[1]), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
            self.draw_bbox(self.right_eye_bbox, (245, 10, 10))
            self.draw_bbox(self.left_eye_bbox, (245, 10, 10))

    def run_pose(self):
        nn_data = run_nn(self.pose_in, self.pose_nn, {"data": to_planar(self.face_frame, (60, 60))})

        self.pose = [val[0] for val in to_tensor_result(nn_data).values()]

        if debug:
            draw_3d_axis(self.debug_frame, self.pose, self.nose)

    def run_gaze(self):
        nn_data = run_nn(self.gaze_in, self.gaze_nn, {
            "lefy_eye_image": to_planar(self.left_eye_image, (60, 60)),
            "right_eye_image": to_planar(self.right_eye_image, (60, 60)),
            "head_pose_angles": self.pose,
        })

        self.gaze = to_nn_result(nn_data)

        if debug:
            re_x = (self.right_eye_bbox[0] + self.right_eye_bbox[2]) // 2
            re_y = (self.right_eye_bbox[1] + self.right_eye_bbox[3]) // 2
            le_x = (self.left_eye_bbox[0] + self.left_eye_bbox[2]) // 2
            le_y = (self.left_eye_bbox[1] + self.left_eye_bbox[3]) // 2

            x, y = (self.gaze * 100).astype(int)[:2]
            cv2.arrowedLine(self.debug_frame, (le_x, le_y), (le_x + x, le_y - y), (255, 0, 255), 3)
            cv2.arrowedLine(self.debug_frame, (re_x, re_y), (re_x + x, re_y - y), (255, 0, 255), 3)

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        face_success = self.run_face()
        if face_success:
            self.run_landmark()
            self.run_pose()
            self.run_gaze()
            print(self.gaze)

        if debug:
            cv2.imshow("Camera_view", cv2.resize(self.debug_frame, (900, 450)))
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
