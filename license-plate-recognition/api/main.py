import argparse
import threading
import time
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np

from depthai_sdk.fps import FPSHandler

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
shaves = 6 if args.camera else 8


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if args.camera:
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(672, 384)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating License Plates Detection Neural Network...")
    det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    det_nn.setConfidenceThreshold(0.5)
    det_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106", shaves=shaves))
    det_nn.input.setQueueSize(1)
    det_nn.input.setBlocking(False)
    det_nn_xout = pipeline.create(dai.node.XLinkOut)
    det_nn_xout.setStreamName("det_nn")
    det_nn.out.link(det_nn_xout.input)

    if args.camera:
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(300, 300)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        cam.preview.link(manip.inputImage)
        manip.out.link(det_nn.input)
    else:
        det_xin = pipeline.create(dai.node.XLinkIn)
        det_xin.setStreamName("det_in")
        det_xin.out.link(det_nn.input)

    # NeuralNetwork
    print("Creating Vehicle Detection Neural Network...")
    veh_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    veh_nn.setConfidenceThreshold(0.5)
    veh_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-adas-0002", shaves=shaves))
    veh_nn.input.setQueueSize(1)
    veh_nn.input.setBlocking(False)
    veh_nn_xout = pipeline.create(dai.node.XLinkOut)
    veh_nn_xout.setStreamName("veh_nn")
    veh_nn.out.link(veh_nn_xout.input)

    if args.camera:
        cam.preview.link(veh_nn.input)
    else:
        veh_xin = pipeline.create(dai.node.XLinkIn)
        veh_xin.setStreamName("veh_in")
        veh_xin.out.link(veh_nn.input)

    rec_nn = pipeline.create(dai.node.NeuralNetwork)
    rec_nn.setBlobPath(blobconverter.from_zoo(name="license-plate-recognition-barrier-0007", shaves=shaves))
    rec_nn.input.setBlocking(False)
    rec_nn.input.setQueueSize(1)
    rec_xout = pipeline.create(dai.node.XLinkOut)
    rec_xout.setStreamName("rec_nn")
    rec_nn.out.link(rec_xout.input)
    rec_pass = pipeline.create(dai.node.XLinkOut)
    rec_pass.setStreamName("rec_pass")
    rec_nn.passthrough.link(rec_pass.input)
    rec_xin = pipeline.create(dai.node.XLinkIn)
    rec_xin.setStreamName("rec_in")
    rec_xin.out.link(rec_nn.input)

    attr_nn = pipeline.create(dai.node.NeuralNetwork)
    attr_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039", shaves=shaves))
    attr_nn.input.setBlocking(False)
    attr_nn.input.setQueueSize(1)
    attr_xout = pipeline.create(dai.node.XLinkOut)
    attr_xout.setStreamName("attr_nn")
    attr_nn.out.link(attr_xout.input)
    attr_pass = pipeline.create(dai.node.XLinkOut)
    attr_pass.setStreamName("attr_pass")
    attr_nn.passthrough.link(attr_pass.input)
    attr_xin = pipeline.create(dai.node.XLinkIn)
    attr_xin.setStreamName("attr_in")
    attr_xin.out.link(attr_nn.input)

    print("Pipeline created.")
    return pipeline


running = True
license_detections = []
vehicle_detections = []
rec_results = []
attr_results = []
frame_det_seq = 0
frame_seq_map = {}
veh_last_seq = 0
lic_last_seq = 0

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)


def lic_thread(det_queue, rec_queue):
    global license_detections, lic_last_seq

    while running:
        try:
            in_det = det_queue.get()
            dets = in_det.detections

            orig_frame = frame_seq_map.get(in_det.getSequenceNum(), None)
            if orig_frame is None:
                continue

            license_detections = [detection for detection in dets if detection.label == 2]

            for detection in license_detections:
                bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setTimestamp(tstamp)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setData(to_planar(cropped_frame, (94, 24)))
                img.setWidth(94)
                img.setHeight(24)
                rec_queue.send(img)

            fps.tick('lic')
        except RuntimeError:
            continue


def veh_thread(det_queue, attr_queue):
    global vehicle_detections, veh_last_seq

    while running:
        try:
            in_dets = det_queue.get()
            vehicle_detections = in_dets.detections

            orig_frame = frame_seq_map.get(in_dets.getSequenceNum(), None)
            if orig_frame is None:
                continue

            veh_last_seq = in_dets.getSequenceNum()

            for detection in vehicle_detections:
                bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setTimestamp(tstamp)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setData(to_planar(cropped_frame, (72, 72)))
                img.setWidth(72)
                img.setHeight(72)
                attr_queue.send(img)

            fps.tick('veh')
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
            rec_data = q_rec.get().getFirstLayerInt32()
            rec_frame = q_pass.get().getCvFrame()
        except RuntimeError:
            continue

        decoded_text = ""
        for idx in rec_data:
            if idx == -1:
                break
            decoded_text += items[int(idx)]
        rec_results = [(cv2.resize(rec_frame, (200, 64)), decoded_text)] + rec_results[:9]
        fps.tick('rec')


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

        fps.tick('attr')


print("Starting pipeline...")
with dai.Device(create_pipeline()) as device:
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        det_in = device.getInputQueue("det_in")
        veh_in = device.getInputQueue("veh_in")
    rec_in = device.getInputQueue("rec_in")
    attr_in = device.getInputQueue("attr_in")
    det_nn = device.getOutputQueue("det_nn", 1, False)
    veh_nn = device.getOutputQueue("veh_nn", 1, False)
    rec_nn = device.getOutputQueue("rec_nn", 1, False)
    rec_pass = device.getOutputQueue("rec_pass", 1, False)
    attr_nn = device.getOutputQueue("attr_nn", 1, False)
    attr_pass = device.getOutputQueue("attr_pass", 1, False)

    det_t = threading.Thread(target=lic_thread, args=(det_nn, rec_in))
    det_t.start()
    veh_t = threading.Thread(target=veh_thread, args=(veh_nn, attr_in))
    veh_t.start()
    rec_t = threading.Thread(target=rec_thread, args=(rec_nn, rec_pass))
    rec_t.start()
    attr_t = threading.Thread(target=attr_thread, args=(attr_nn, attr_pass))
    attr_t.start()


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        global frame_det_seq

        if args.video:
            read_correctly, frame = cap.read()
            if read_correctly:
                frame_seq_map[frame_det_seq] = frame
                frame_det_seq += 1
            return read_correctly, frame
        else:
            in_rgb = cam_out.get()
            frame = in_rgb.getCvFrame()
            frame_seq_map[in_rgb.getSequenceNum()] = frame

            return True, frame


    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break

            for map_key in list(filter(lambda item: item <= min(lic_last_seq, veh_last_seq), frame_seq_map.keys())):
                del frame_seq_map[map_key]

            fps.nextIter()

            if not args.camera:
                tstamp = time.monotonic()
                lic_frame = dai.ImgFrame()
                lic_frame.setData(to_planar(frame, (300, 300)))
                lic_frame.setTimestamp(tstamp)
                lic_frame.setSequenceNum(frame_det_seq)
                lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
                lic_frame.setWidth(300)
                lic_frame.setHeight(300)
                det_in.send(lic_frame)
                veh_frame = dai.ImgFrame()
                veh_frame.setData(to_planar(frame, (300, 300)))
                veh_frame.setTimestamp(tstamp)
                veh_frame.setSequenceNum(frame_det_seq)
                veh_frame.setType(dai.RawImgFrame.Type.BGR888p)
                veh_frame.setWidth(300)
                veh_frame.setHeight(300)
                veh_frame.setData(to_planar(frame, (672, 384)))
                veh_frame.setWidth(672)
                veh_frame.setHeight(384)
                veh_in.send(veh_frame)

            if debug:
                debug_frame = frame.copy()
                for detection in license_detections + vehicle_detections:
                    bbox = frame_norm(debug_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(debug_frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0))
                cv2.putText(debug_frame, f"LIC FPS:  {round(fps.tickFps('lic'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"VEH FPS:  {round(fps.tickFps('veh'), 1)}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"REC FPS:  {round(fps.tickFps('rec'), 1)}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.putText(debug_frame, f"ATTR FPS:  {round(fps.tickFps('attr'), 1)}", (5, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0))
                cv2.imshow("rgb", debug_frame)

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
                    cv2.putText(attr_placeholder_img, f"{int(color_prob * 100)}%", (150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(type_prob * 100)}%", (150, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0))
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
    veh_t.join()

print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
