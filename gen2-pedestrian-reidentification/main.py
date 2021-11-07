import argparse
import time
import queue
import signal
import threading
from pathlib import Path

import cv2
import depthai as dai
print('depthai module: ', dai.__file__)
import numpy as np
from imutils.video import FPS

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-w', '--width', default=1280, type=int, help="Visualization width. Height is calculated automatically from aspect ratio")
parser.add_argument('-lq', '--lowquality', action="store_true", help="Low quality visualization - uses resized frames")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug
camera = not args.video
hq = not args.lowquality

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is False and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(544, 320)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        # Link video output to host for higher resolution
        if hq:
            cam.video.link(cam_xout.input)
        else:
            cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Person Detection Neural Network...")
    detection_nn = pipeline.createMobileNetDetectionNetwork()
    detection_nn.setBlobPath(str(Path("models/person-detection-retail-0013_openvino_2020.1_4shave.blob").resolve().absolute()))
    # Confidence
    detection_nn.setConfidenceThreshold(0.7)
    # Increase threads for detection
    detection_nn.setNumInferenceThreads(2)
    # Specify that network takes latest arriving frame in non-blocking manner
    detection_nn.input.setQueueSize(1)
    detection_nn.input.setBlocking(False)

    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")

    detection_nn_passthrough = pipeline.createXLinkOut()
    detection_nn_passthrough.setStreamName("detection_passthrough")
    detection_nn_passthrough.setMetadataOnly(True)

    if camera:
        print('linked cam.preview to detection_nn.input')
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    detection_nn.out.link(detection_nn_xout.input)
    detection_nn.passthrough.link(detection_nn_passthrough.input)


    # NeuralNetwork
    print("Creating Person Reidentification Neural Network...")
    reid_in = pipeline.createXLinkIn()
    reid_in.setStreamName("reid_in")
    reid_nn = pipeline.createNeuralNetwork()
    reid_nn.setBlobPath(str(Path("models/person-reidentification-retail-0031_openvino_2020.1_4shave.blob").resolve().absolute()))
    
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

        self.running = True
        self.FRAMERATE = 30.0

        if not camera:
            self.cap = cv2.VideoCapture(args.video)
            self.FRAMERATE = self.cap.get(cv2.CAP_PROP_FPS)

        print("framerate: ", self.FRAMERATE)

        self.frame_queue = queue.Queue()
        self.visualization_queue = queue.Queue(maxsize=4)

        self.nn_fps = 0

    def is_running(self):
        if self.running:
            if camera:
                return True
            else:
                return self.cap.isOpened()
        return False

    def inference_task(self):
        # Queues
        detection_passthrough = self.device.getOutputQueue("detection_passthrough")
        detection_nn = self.device.getOutputQueue("detection_nn")

        bboxes = []
        results = {}
        results_path = {}
        next_id = 0

        # Match up frames and detections
        try:
            prev_passthrough = detection_passthrough.getAll()[0]
            prev_inference = detection_nn.getAll()[0]
        except RuntimeError:
            pass

        fps = 0
        t_fps = time.time()
        while self.running:
            try:

                # Get current detection
                passthrough = detection_passthrough.getAll()[0]
                inference = detection_nn.getAll()[0]

                # Count NN fps
                fps = fps + 1

                # Combine all frames to current inference
                frames = []
                while True:

                    frm = self.frame_queue.get()
                    if camera and hq:
                        # Convert NV12 to BGR
                        yuv = frm.getData().reshape((frm.getHeight() * 3 // 2, frm.getWidth()))
                        cv_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)                      
                    else:
                        # get the frames corresponding to inference
                        cv_frame = np.ascontiguousarray(frm.getData().reshape(3, frm.getHeight(), frm.getWidth()).transpose(1,2,0))

                    frames.append(cv_frame)

                    # Break out once all frames received for the current inference
                    if frm.getSequenceNum() >= prev_passthrough.getSequenceNum() - 1:
                        break

                infered_frame = frames[0]

                # Send bboxes to be infered upon
                for det in inference.detections:
                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)
                    det_frame = infered_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    nn_data = dai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    self.device.getInputQueue("reid_in").send(nn_data)

                 
                # Retrieve infered bboxes
                for det in inference.detections:

                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)

                    reid_result = self.device.getOutputQueue("reid_nn").get().getFirstLayerFp16()

                    for person_id in results:
                        dist = cos_dist(reid_result, results[person_id])
                        if dist > 0.7:
                            result_id = person_id
                            results[person_id] = reid_result
                            break
                    else:
                        result_id = next_id
                        results[result_id] = reid_result
                        results_path[result_id] = []
                        next_id += 1

                    if debug:
                        for frame in frames:
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                            x = (bbox[0] + bbox[2]) // 2
                            y = (bbox[1] + bbox[3]) // 2
                            results_path[result_id].append([x, y])
                            cv2.putText(frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                            if len(results_path[result_id]) > 1:
                                cv2.polylines(frame, [np.array(results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)
                    else:
                        print(f"Saw id: {result_id}")

                # Send of to visualization thread
                for frame in frames:
                    # put nn_fps
                    if debug:
                        cv2.putText(frame, 'NN FPS: '+str(self.nn_fps), (5,40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 2)

                    if self.visualization_queue.full():
                        self.visualization_queue.get_nowait()
                    self.visualization_queue.put(frame)
            

                # Move current to prev
                prev_passthrough = passthrough
                prev_inference = inference

                if time.time() - t_fps >= 1.0:
                    self.nn_fps = round(fps / (time.time() - t_fps), 2)
                    fps = 0
                    t_fps = time.time()

            except RuntimeError:
                continue



    def input_task(self):
        seq_num = 0
        while self.is_running():

            # Send images to next stage

            # if camera - receive frames from camera
            if camera:
                try:
                    frame = self.device.getOutputQueue('cam_out').get()
                    self.frame_queue.put(frame)
                except RuntimeError:
                    continue
            
            # else if video - send frames down to NN
            else:

                # Get frame from video capture
                read_correctly, vid_frame = self.cap.read()
                if not read_correctly:
                    break

                # Send to NN and to inference thread
                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(seq_num)
                frame_nn.setWidth(544)
                frame_nn.setHeight(320)
                frame_nn.setData(to_planar(vid_frame, (544, 320)))
                self.device.getInputQueue("detection_in").send(frame_nn)

                # if high quality, send original frames
                if hq:
                    frame_orig = dai.ImgFrame()
                    frame_orig.setSequenceNum(seq_num)
                    frame_orig.setWidth(vid_frame.shape[1])
                    frame_orig.setHeight(vid_frame.shape[0])
                    frame_orig.setData(to_planar(vid_frame, (vid_frame.shape[1], vid_frame.shape[0])))
                    self.frame_queue.put(frame_orig)
                # else send resized frames
                else: 
                    self.frame_queue.put(frame_nn)                

                seq_num = seq_num + 1

                # Sleep at video framerate
                time.sleep(1.0 / self.FRAMERATE)
        # Stop execution after input task doesn't have
        # any extra data anymore
        self.running = False


    def visualization_task(self):
        
        first = True
        while self.running:

            t1 = time.time()

            # Show frame if available
            if first or not self.visualization_queue.empty():
                frame = self.visualization_queue.get()
                aspect_ratio = frame.shape[1] / frame.shape[0]
                cv2.imshow("frame", cv2.resize(frame, (int(args.width),  int(args.width / aspect_ratio))))
                first = False

            # sleep if required
            to_sleep_ms = ((1.0 / self.FRAMERATE) - (time.time() - t1)) * 1000
            key = None
            if to_sleep_ms >= 1:
                key = cv2.waitKey(int(to_sleep_ms)) 
            else:
                key = cv2.waitKey(1)
            # Exit
            if key == ord('q'):
                self.running = False
                break

            

    def run(self):


        pipeline = create_pipeline()

        # Connect to the device
        print("Starting pipeline...")
        with dai.Device(pipeline) as device:
            self.device = device

            threads = [
                threading.Thread(target=self.input_task),
                threading.Thread(target=self.inference_task),
            ]
            for t in threads:
                t.start()

            # Visualization task should run in 'main' thread
            self.visualization_task()

            # cleanup
        self.running = False
        if not camera:
            self.cap.release()
        for thread in threads:
            thread.join()


# Create the application
app = Main()

# Register a graceful CTRL+C shutdown
def signal_handler(sig, frame):
    app.running = False
signal.signal(signal.SIGINT, signal_handler)

# Run the application
app.run()
# Print latest NN FPS
print('FPS: ', app.nn_fps)