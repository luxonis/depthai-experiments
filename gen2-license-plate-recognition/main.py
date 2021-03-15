import argparse
import threading
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Use either \"-cam\" to run on RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if args.camera:
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(300, 300)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Detection Neural Network...")
    det_nn = pipeline.createMobileNetDetectionNetwork()
    det_nn.setConfidenceThreshold(0.5)
    det_nn.setBlobPath(str(Path("models/vehicle-license-plate-detection-barrier-0106.blob").resolve().absolute()))
    det_nn.setNumInferenceThreads(2)
    det_nn.input.setQueueSize(1)
    det_nn.input.setBlocking(False)
    det_nn_xout = pipeline.createXLinkOut()
    det_nn_xout.setStreamName("det_nn")
    det_nn.out.link(det_nn_xout.input)
    det_pass = pipeline.createXLinkOut()
    det_pass.setStreamName("det_pass")
    det_nn.passthrough.link(det_pass.input)

    if args.camera:
        cam.preview.link(det_nn.input)
    else:
        det_xin = pipeline.createXLinkIn()
        det_xin.setStreamName("det_in")
        det_xin.out.link(det_nn.input)

    rec_nn = pipeline.createNeuralNetwork()
    rec_nn.setBlobPath(str((Path(__file__).parent / Path('models/license-plate-recognition-barrier-0007.blob')).resolve().absolute()))
    rec_nn.input.setBlocking(False)
    rec_nn.setNumInferenceThreads(2)
    rec_nn.input.setQueueSize(1)
    rec_xout = pipeline.createXLinkOut()
    rec_xout.setStreamName("rec_nn")
    rec_nn.out.link(rec_xout.input)
    rec_pass = pipeline.createXLinkOut()
    rec_pass.setStreamName("rec_pass")
    rec_nn.passthrough.link(rec_pass.input)
    rec_xin = pipeline.createXLinkIn()
    rec_xin.setStreamName("rec_in")
    rec_xin.out.link(rec_nn.input)

    attr_nn = pipeline.createNeuralNetwork()
    attr_nn.setBlobPath(str((Path(__file__).parent / Path('models/vehicle-attributes-recognition-barrier-0039.blob')).resolve().absolute()))
    attr_nn.input.setBlocking(False)
    attr_nn.setNumInferenceThreads(2)
    attr_nn.input.setQueueSize(1)
    attr_xout = pipeline.createXLinkOut()
    attr_xout.setStreamName("attr_nn")
    attr_nn.out.link(attr_xout.input)
    attr_pass = pipeline.createXLinkOut()
    attr_pass.setStreamName("attr_pass")
    attr_nn.passthrough.link(attr_pass.input)
    attr_xin = pipeline.createXLinkIn()
    attr_xin.setStreamName("attr_in")
    attr_xin.out.link(attr_nn.input)

    print("Pipeline created.")
    return pipeline


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


running = True
detections = []
rec_results = []
attr_results = []
licese_frame = None
frame_det_seq = 0
frame_det_seq_map = {}

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)


def det_thread(det_queue, det_pass, rec_queue, attr_queue):
    global detections, licese_frame

    while running:
        try:
            detections = det_queue.get().detections
            in_pass = det_pass.get()
            orig_frame = frame_det_seq_map.get(in_pass.getSequenceNum(), None)
            if orig_frame is None:
                print("No matching frame found, skipping...")
                continue

            for map_key in list(filter(lambda item: item <= in_pass.getSequenceNum(), frame_det_seq_map.keys())):
                del frame_det_seq_map[map_key]

            for detection in detections:
                bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setTimestamp(tstamp)
                img.setType(dai.RawImgFrame.Type.BGR888p)

                if detection.label == 2:  # license plate
                    img.setData(to_planar(cropped_frame, (94, 24)))
                    img.setWidth(94)
                    img.setHeight(24)
                    rec_queue.send(img)
                elif detection.label == 1:  # car
                    img.setData(to_planar(cropped_frame, (72, 72)))
                    img.setWidth(72)
                    img.setHeight(72)
                    attr_queue.send(img)

            fps.tick('det')
        except RuntimeError:
            continue


items = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>", "<Gansu>",
         "<Guangdong>", "<Guangxi>", "<Guizhou>", "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>", "<HongKong>",
         "<Hubei>", "<Hunan>", "<InnerMongolia>", "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>", "<Macau>",
         "<Ningxia>", "<Qinghai>", "<Shaanxi>", "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>", "<Tianjin>",
         "<Tibet>", "<Xinjiang>", "<Yunnan>", "<Zhejiang>", "<police>", "A", "B", "C", "D", "E", "F", "G", "H", "I",
         "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def rec_thread(q_rec, q_pass):
    global rec_results

    while running:
        try:
            rec_data = q_rec.get().getFirstLayerFp16()
            rec_frame = q_pass.get().getCvFrame()
        except RuntimeError:
            continue

        decoded_text = ""
        for idx in rec_data:
            if idx == -1:
                break
            decoded_text += items[int(idx)]
        rec_results = [(cv2.resize(rec_frame, (200, 64)), decoded_text)] + rec_results[:9]
        fps.tick_fps('rec')


def attr_thread(q_attr, q_pass):
    global attr_results

    while running:
        try:
            attr_data = q_attr.get()
            attr_frame = q_pass.get().getCvFrame()
        except RuntimeError:
            continue

        colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
        types = ["car", "bus", "truck", "van"]

        in_color = np.array(attr_data.getLayerFp16("color"))
        in_type = np.array(attr_data.getLayerFp16("type"))

        color = colors[in_color.argmax()]
        color_prob = float(in_color.max())
        type = types[in_type.argmax()]
        type_prob = float(in_type.max())

        attr_results = [(attr_frame, color, type, color_prob, type_prob)] + attr_results[:9]

        fps.tick_fps('attr')


with dai.Device(create_pipeline()) as device:
    print("Starting pipeline...")
    device.startPipeline()
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        det_in = device.getInputQueue("det_in")
    rec_in = device.getInputQueue("rec_in")
    attr_in = device.getInputQueue("attr_in")
    det_nn = device.getOutputQueue("det_nn", 1, False)
    det_pass = device.getOutputQueue("det_pass", 1, False)
    rec_nn = device.getOutputQueue("rec_nn", 1, False)
    rec_pass = device.getOutputQueue("rec_pass", 1, False)
    attr_nn = device.getOutputQueue("attr_nn", 1, False)
    attr_pass = device.getOutputQueue("attr_pass", 1, False)

    det_t = threading.Thread(target=det_thread, args=(det_nn, det_pass, rec_in, attr_in))
    det_t.start()
    rec_t = threading.Thread(target=rec_thread, args=(rec_nn, rec_pass))
    rec_t.start()
    attr_t = threading.Thread(target=attr_thread, args=(attr_nn, attr_pass))
    attr_t.start()


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        if args.video:
            return cap.read()
        else:
            return True, cam_out.get().getCvFrame()


    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break

            fps.next_iter()

            if not args.camera:
                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setData(to_planar(frame, (300, 300)))
                img.setTimestamp(tstamp)
                img.setSequenceNum(frame_det_seq)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setWidth(300)
                img.setHeight(300)
                det_in.send(img)
                frame_det_seq_map[frame_det_seq] = frame
                frame_det_seq += 1

            if debug:
                debug_frame = frame.copy()
                for detection in detections:
                    bbox = frame_norm(debug_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(debug_frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0))
                cv2.putText(debug_frame, f"NN FPS:  {round(fps.tick_fps('det'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"REC FPS:  {round(fps.tick_fps('rec'), 1)}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"ATTR FPS:  {round(fps.tick_fps('attr'), 1)}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.imshow("rgb", debug_frame)

                if licese_frame is not None:
                    cv2.imshow("license plate", licese_frame)

                rec_stacked = None

                for rec_img, rec_text in rec_results:
                    rec_placeholder_img = np.zeros((64, 200, 3), np.uint8)
                    cv2.putText(rec_placeholder_img, rec_text, (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    rec_combined = np.hstack((rec_img, rec_placeholder_img))

                    if rec_stacked is None:
                        rec_stacked = rec_combined
                    else:
                        rec_stacked = np.vstack((rec_stacked, rec_combined))

                if rec_stacked is not None:
                    cv2.imshow("Recognized plates", rec_stacked)

                attr_stacked = None

                for attr_img, attr_color, attr_type, color_prob, type_prob in attr_results:
                    attr_placeholder_img = np.zeros((72, 200, 3), np.uint8)
                    cv2.putText(attr_placeholder_img, attr_color, (15, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, attr_type, (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(color_prob * 100)}%", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(type_prob * 100)}%", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    attr_combined = np.hstack((attr_img, attr_placeholder_img))

                    if attr_stacked is None:
                        attr_stacked = attr_combined
                    else:
                        attr_stacked = np.vstack((attr_stacked, attr_combined))

                if attr_stacked is not None:
                    cv2.imshow("Attributes", attr_stacked)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    running = False

det_t.join()
rec_t.join()
attr_t.join()
print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
