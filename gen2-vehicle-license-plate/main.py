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


texts = ["", "vehicle", "license-plate", "", "", "", "", "", "", "", "",
         "", "", "", "", "", "", "", "", "", ""]

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
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

    if camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        # cam.setPreviewSize(300, 300)
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
    print("Creating Vehicle license plate Detection Neural Network...")
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("models/vehicle-license-plate-detection-barrier-0106.2021.2.4shave4slice.blob").resolve().absolute()))

    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")

    detection_nn_passthrough = pipeline.createXLinkOut()
    detection_nn_passthrough.setStreamName("detection_passthrough")
    detection_nn_passthrough.setMetadataOnly(True)

    if camera:
        print('linked cam.preview to detection_nn.input')
        cam.preview.link(detection_nn.input)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")

        cam.preview.link(xout_rgb.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        detection_in.out.link(xout_rgb.input)

    detection_nn.out.link(detection_nn_xout.input)
    detection_nn.passthrough.link(detection_nn_passthrough.input)

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
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = self.device.getOutputQueue(name="detection_nn", maxSize=4, blocking=False)

        frame = None
        bboxes = []
        confidences = []
        labels = []
        start_time = time.time()
        counter = 0
        fps = 0

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frame_norm(frame, bbox):
            norm_vals = np.full(len(bbox), frame.shape[0])
            norm_vals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


        while True:
            # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)

            if in_nn is not None:
                # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
                bboxes = np.array(in_nn.getFirstLayerFp16())
                # transform the 1D array into Nx7 matrix
                bboxes = bboxes.reshape((bboxes.size // 7, 7))
                # filter out the results which confidence less than a defined threshold
                bboxes = bboxes[bboxes[:, 2] > 0.3]
                # Cut bboxes and labels
                labels = bboxes[:, 1].astype(int)
                confidences = bboxes[:, 2]
                bboxes = bboxes[:, 3:7]
                counter+=1
                current_time = time.time()
                if (current_time - start_time) > 1 :
                    fps = counter / (current_time - start_time)
                    counter = 0
                    start_time = current_time
                print("bboxes: %r", bboxes)
                print("labels: %r", labels)
                print("confidences: %r", confidences)

            color = (255, 255, 255)

            if frame is not None:
                # if the frame is available, draw bounding boxes on it and show the frame
                for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                    bbox = frame_norm(frame, raw_bbox)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(frame, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                    if self.visualization_queue.full():
                        self.visualization_queue.get_nowait()
                    self.visualization_queue.put(frame)


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
                frame_nn.setWidth(300)
                frame_nn.setHeight(300)
                frame_nn.setData(to_planar(vid_frame, (300, 300)))
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
                cv2.imshow("rgb", cv2.resize(frame, (int(args.width),  int(args.width / aspect_ratio))))
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
        with dai.Device(pipeline) as device:
            self.device = device

            print("Starting pipeline...")
            device.startPipeline()            

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
