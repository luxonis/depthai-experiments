import argparse
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
import depthai
import time
from tools import *
from imutils.video import FPS
import queue
import threading

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


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def coordinate(frame, *xy_vals):
    height, width = frame.shape[:2]
    result = []
    for i, val in enumerate(xy_vals):
        if i % 2 == 0:
            result.append(max(0, min(width, int(val * width))))
        else:
            result.append(max(0, min(height, int(val * height))))
    return result

def create_pipeline():
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
        first_models(cam,pipeline,"models/face-detection-retail-0004_openvino_2020_1_4shave.blob","face")
    else:
        models(pipeline,"models/face-detection-retail-0004_openvino_2020_1_4shave.blob","face")
    models(pipeline,"models/face_landmark_160x160_openvino_2020_1_4shave.blob","land68")
    return pipeline

def first_models(cam,pipeline,model_path,name):
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
    def __init__(self):
        print("Loading pipeline...")
        self.start_pipeline()
        if not camera:
            self.cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        self.fps = FPS()
        self.fps.start()
        self.frame = None
        self.face_box_q = queue.Queue()
        self.bboxes = []
        self.running = True
        self.result = None
        self.face_bbox = None

    def start_pipeline(self):
        print("Starting pipeline...")
        self.device = depthai.Device(create_pipeline())
        if camera:
            self.cam_out = self.device.getOutputQueue("cam_out",1,False)
        else:
            self.face_in = self.device.getInputQueue("face_in")

    def draw_bbox(self, bbox, color):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def run_face(self):
        face_nn = self.device.getOutputQueue("face_nn")
        land68_in = self.device.getInputQueue("land68_in",4,False)
        while self.running:
            if self.frame is None:
                continue
            bboxes = np.array(face_nn.get().getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            self.bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]
            for raw_bbox in self.bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                land68_data = depthai.NNData()
                land68_data.setLayer("data", to_planar(det_frame, (160,160)))
                land68_in.send(land68_data)
                self.face_box_q.put(bbox)

    def run_land68(self):
        land68_nn = self.device.getOutputQueue("land68_nn",4,False)

        while self.running:
            self.results = queue.Queue()
            self.face_bboxs = queue.Queue()
            while len(self.bboxes):
                face_bbox = self.face_box_q.get()
                face_bbox[1] -= 15
                face_bbox[3] += 15
                face_bbox[0] -= 15
                face_bbox[2] += 15
                self.face_bboxs.put(face_bbox)
                face_frame = self.frame[
                    face_bbox[1]:face_bbox[3],
                    face_bbox[0]:face_bbox[2]
                ]
                land68_data = land68_nn.get()
                out = to_tensor_result(land68_data).get('StatefulPartitionedCall/strided_slice_2/Split.0')
                result = coordinate(face_frame,*out)
                self.results.put(result)

    def should_run(self):
        return True if camera else self.cap.isOpened()

    def get_frame(self, retries=0):
        if camera:
            rgb_output = self.cam_out.get()
            return True, np.array(rgb_output.getData()).reshape((3, rgb_output.getHeight(), rgb_output.getWidth())).transpose(1, 2, 0).astype(np.uint8)
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
            threading.Thread(target=self.run_face, daemon=True),
            threading.Thread(target=self.run_land68, daemon=True)
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
                nn_data.setLayer("data", to_planar(self.frame, (300, 300)))
                self.face_in.send(nn_data)
            if debug:
                if self.results.qsize() > 0 and self.face_bboxs.qsize() > 0:
                    try:
                        for i in range(self.results.qsize()):
                            face_bbox = self.face_bboxs.get()
                            result = self.results.get()
                            bbox = frame_norm(self.frame, self.bboxes[i])
                            self.draw_bbox(bbox, (0,255,0))
                            self.hand_points = []
                            # 17 Left eyebrow upper left corner/21 Left eyebrow right corner/22 Right eyebrow upper left corner/26 Right eyebrow upper right corner/36 Left eye upper left corner/39 Left eye upper right corner/42 Right eye upper left corner/
                            # 45 Upper right corner of the right eye/31 Upper left corner of the nose/35 Upper right corner of the nose/48 Upper left corner/54 Upper right corner of the mouth/57 Lower central corner of the mouth/8 Chin corner
                            # The coordinates are two points, so you have to multiply by 2.
                            self.hand_points.append((result[34]+face_bbox[0],result[35]+face_bbox[1]))
                            self.hand_points.append((result[42]+face_bbox[0],result[43]+face_bbox[1]))
                            self.hand_points.append((result[44]+face_bbox[0],result[45]+face_bbox[1]))
                            self.hand_points.append((result[52]+face_bbox[0],result[53]+face_bbox[1]))
                            self.hand_points.append((result[72]+face_bbox[0],result[73]+face_bbox[1]))
                            self.hand_points.append((result[78]+face_bbox[0],result[79]+face_bbox[1]))
                            self.hand_points.append((result[84]+face_bbox[0],result[85]+face_bbox[1]))
                            self.hand_points.append((result[90]+face_bbox[0],result[91]+face_bbox[1]))
                            self.hand_points.append((result[62]+face_bbox[0],result[63]+face_bbox[1]))
                            self.hand_points.append((result[70]+face_bbox[0],result[71]+face_bbox[1]))
                            self.hand_points.append((result[96]+face_bbox[0],result[97]+face_bbox[1]))
                            self.hand_points.append((result[108]+face_bbox[0],result[109]+face_bbox[1]))
                            self.hand_points.append((result[114]+face_bbox[0],result[115]+face_bbox[1]))
                            self.hand_points.append((result[16]+face_bbox[0],result[17]+face_bbox[1]))
                            for i in self.hand_points:
                                cv2.circle(self.debug_frame,(i[0],i[1]),2,(255,0,0),thickness=1,lineType=8,shift=0)
                            reprojectdst, _, pitch, yaw, roll = get_head_pose(np.array(self.hand_points))

                            """
                            pitch > 0 Head down, < 0 look up
                            yaw > 0 Turn right < 0 Turn left
                            roll > 0 Tilt right, < 0 Tilt left
                            """
                            cv2.putText(self.debug_frame,"pitch:{:.2f}, yaw:{:.2f}, roll:{:.2f}".format(pitch,yaw,roll),(face_bbox[0]-30,face_bbox[1]-30),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,0,0))  
                            
                            hand_attitude = np.array([abs(pitch),abs(yaw),abs(roll)])
                            max_index = np.argmax(hand_attitude)
                            if max_index == 0:
                                if pitch > 0:
                                    cv2.putText(self.debug_frame,"Head down", (face_bbox[0],face_bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(235,10,10))
                                else:
                                    cv2.putText(self.debug_frame,"look up", (face_bbox[0],face_bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(235,10,10))
                            elif max_index == 1:
                                if yaw > 0:
                                    cv2.putText(self.debug_frame,"Turn right", (face_bbox[0],face_bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(235,10,10))
                                else:
                                    cv2.putText(self.debug_frame,"Turn left", (face_bbox[0],face_bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(235,10,10))
                            elif max_index == 2:
                                if roll > 0:
                                    cv2.putText(self.debug_frame,"Tilt right", (face_bbox[0],face_bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(235,10,10))
                                else:
                                    cv2.putText(self.debug_frame,"Tilt left", (face_bbox[0],face_bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(235,10,10))
                            # Draw a cube with 12 axes
                            line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                                        [4, 5], [5, 6], [6, 7], [7, 4],
                                        [0, 4], [1, 5], [2, 6], [3, 7]]
                            for start, end in line_pairs:
                                cv2.line(self.debug_frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                    except:
                        pass
                if camera:
                    cv2.imshow("Camera view", self.debug_frame)
                else:
                    aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
                    cv2.imshow("Video view", cv2.resize(self.debug_frame, (int(900),  int(900 / aspect_ratio))))
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

        self.fps.stop()
        print("FPS：{:.2f}".format(self.fps.fps()))
        if not camera:
            self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        for thread in self.threads:
            thread.join(2)
            if thread.is_alive():
                break

if __name__ == '__main__':
    Main().run()