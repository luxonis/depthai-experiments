import argparse
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
import depthai
from scipy.spatial.distance import euclidean
import os
import time
from tools import *
from imutils.video import FPS

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

def timer(function):
    """
    Decorator function timer
    :param function:The function you want to time
    :return:
    """

    def wrapper(*args, **kwargs):
        time_start = time.time()
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        print("【 %s 】operation hours：【 %s 】second" % (function.__name__, cost_time))
        return res

    return wrapper

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
    try:
        arr = to_nn_result(nn_data)
        arr = arr[:np.where(arr == -1)[0][0]]
        arr = arr.reshape((arr.size // 7, 7))
        return arr
    except:
        return []

#@timer
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

def create_pipeline(camera):
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()
    if camera:
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(300,300)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)
        first_model(pipeline,cam,"models/face-detection-retail-0004_openvino_2020_1_4shave.blob","face")
    else:
        models(pipeline,"models/face-detection-retail-0004_openvino_2020_1_4shave.blob","face")
    models(pipeline,"models/face_landmark_160x160_openvino_2020_1_4shave.blob","land68")
    return pipeline

def first_model(pipeline,cam,model_path,name):
    print(f"Start creating{model_path}Neural Networks")
    model_nn = pipeline.createNeuralNetwork()
    model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
    cam.preview.link(model_nn.input)
    model_nn_xout = pipeline.createXLinkOut()
    model_nn_xout.setStreamName(f"{name}_nn")
    model_nn.out.link(model_nn_xout.input)

def models(pipeline,model_path,name):
    print(f"Start creating{model_path}Neural Networks")
    model_in = pipeline.createXLinkIn()
    model_in.setStreamName(f"{name}_in")
    model_nn = pipeline.createNeuralNetwork()
    model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
    model_nn_xout = pipeline.createXLinkOut()
    model_nn_xout.setStreamName(f"{name}_nn")
    model_in.out.link(model_nn.input)
    model_nn.out.link(model_nn_xout.input)

class Main:
    def __init__(self,file=None,camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.pipeline = create_pipeline(self.camera)
        self.start_pipeline()
        self.COUNTER = 0
        self.mCOUNTER = 0
        self.hCOUNTER = 0
        self.TOTAL = 0
        self.mTOTAL = 0
        self.hTOTAL = 0
        self.fps = FPS()
        self.fps.start()

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")
        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 4, False)
        else:
            self.face_in = self.device.getInputQueue("face_in",4,False)
        self.face_nn = self.device.getOutputQueue("face_nn",4,False)
        self.land68_in = self.device.getInputQueue("land68_in",4,False)
        self.land68_nn = self.device.getOutputQueue("land68_nn",4,False)

    def draw_bbox(self, bbox, color):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def run_face(self):
        if self.camera:
            nn_data = self.face_nn.tryGet()
        else:
            nn_data = run_nn(self.face_in, self.face_nn, {"data": to_planar(self.frame, (300, 300))})
        results = to_bbox_result(nn_data)
        self.face_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.4
        ]
        if len(self.face_coords) == 0:
            return False
        if len(self.face_coords) > 0:
            for face_coord in self.face_coords:
                face_coord[0] -= 15
                face_coord[1] -= 15
                face_coord[2] += 15
                face_coord[3] += 15
            self.face_frame = [self.frame[
                face_coord[1]:face_coord[3],
                face_coord[0]:face_coord[2]
            ] for face_coord in self.face_coords]
        if debug:  
            for bbox in self.face_coords:
                self.draw_bbox(bbox, (10, 245, 10))
        return True
    
    
    def run_land68(self,face_frame,count):
        try:
            nn_data = run_nn(self.land68_in,self.land68_nn, {"data": to_planar(face_frame, (160,160))})
            out = to_nn_result(nn_data)
            result = frame_norm(face_frame,*out)
            eye_left = []
            eye_right = []
            mouth = []
            hand_points = []
            for i in range(72,84,2):
                eye_left.append((out[i],out[i+1]))
            for i in range(84,96,2):
                eye_right.append((out[i],out[i+1]))
            for i in range(96,len(result),2):
                if i == 100 or i == 116 or i == 104 or i == 112 or i == 96 or i == 108:
                    mouth.append(np.array([out[i],out[i+1]]))

            for i in range(16,110,2):
                if i == 16 or i == 60 or i == 72 or i == 90 or i == 96 or i == 108:
                    cv2.circle(self.debug_frame,(result[i]+self.face_coords[count][0],result[i+1]+self.face_coords[count][1]),2,(255,0,0),thickness=1,lineType=8,shift=0)
                    hand_points.append((result[i]+self.face_coords[count][0],result[i+1]+self.face_coords[count][1]))
            

            ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(self.frame.shape, np.array(hand_points,dtype='double'))
            ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            p1 = ( int(hand_points[1][0]), int(hand_points[1][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(self.debug_frame, p1, p2, (255,0,0), 2)
            euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
            cv2.putText(self.debug_frame,euler_angle_str,(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,0,255))
            if pitch < 0:
                self.hCOUNTER += 1
                if self.hCOUNTER >= 20:
                    cv2.putText(self.debug_frame,"SLEEP!!!",(self.face_coords[count][0],self.face_coords[count][1]-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
            else:
                if self.hCOUNTER >= 3:
                    self.hTOTAL += 1
                self.hCOUNTER = 0
            
            left_ear = self.eye_aspect_ratio(eye_left)
            right_ear = self.eye_aspect_ratio(eye_right)
            ear = (left_ear + right_ear) / 2.0
            if ear < 0.15:# Eye aspect ratio：0.15
                self.COUNTER += 1
                if self.COUNTER >= 20:
                    cv2.putText(self.debug_frame, "SLEEP!!!", (self.face_coords[count][0], self.face_coords[count][1]-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # If it is less than the threshold 3 times in a row, it means that an eye blink has been performed
                if self.COUNTER >= 3:# Threshold: 3
                    self.TOTAL += 1
                # Reset the eye frame counter
                self.COUNTER = 0

            mouth_ratio = self.mouth_aspect_ratio(mouth)
            if mouth_ratio > 0.5:
                self.mCOUNTER += 1
            else:
                if self.mCOUNTER >= 3:
                    self.mTOTAL += 1
                self.mCOUNTER = 0

            cv2.putText(self.debug_frame,"eye:{:d},mouth:{:d},head:{:d}".format(self.TOTAL,self.mTOTAL,self.hTOTAL),(10,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,0,0,))
            if self.TOTAL >= 50 or self.mTOTAL>=15 or self.hTOTAL >= 10:
                cv2.putText(self.debug_frame, "Danger!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        except:
            pass
    
    def mouth_aspect_ratio(self,mouth):
        A = np.linalg.norm(mouth[1] - mouth[5])  # 51, 59
        B = np.linalg.norm(mouth[2] - mouth[4])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[3])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def eye_aspect_ratio(self,eye):
        A = euclidean(eye[1],eye[5])
        B = euclidean(eye[2],eye[4])
        C = euclidean(eye[0],eye[3])
        return (A + B) / (2.0 * C)

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        face_success = self.run_face()
        if face_success:
            for i in range(len(self.face_frame)):
                self.run_land68(self.face_frame[i],i)
            self.fps.update()
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
            rgb_in = self.cam_out.get()
            self.frame = np.array(rgb_in.getData()).reshape((3, rgb_in.getHeight(), rgb_in.getWidth())).transpose(1, 2, 0).astype(np.uint8)
            try:
                self.parse()
            except StopIteration:
                break
    
    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        self.fps.stop()
        print("FPS:{:.2f}".format(self.fps.fps()))
        del self.device

if __name__ == '__main__':
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()