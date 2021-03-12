#!/usr/bin/env python3
import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import depthai
import numpy as np
import depthai as dai


parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug

pipeline = dai.Pipeline()

if args.camera:
    colorCam = pipeline.createColorCamera()
    colorCam.setPreviewSize(300, 300)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("preview")
    colorCam.preview.link(cam_xout.input)

nn = pipeline.createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(str((Path(__file__).parent / Path('models/vehicle-license-plate-detection-barrier-0106.blob')).resolve().absolute()))
if args.camera:
    colorCam.preview.link(nn.input)
else:
    nn_in = pipeline.createXLinkIn()
    nn_in.setStreamName("nn_in")
    nn_in.out.link(nn.input)

cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("detections_pass")
nn.passthrough.link(cam_xout.input)

nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("detections")
nn.out.link(nn_xout.input)

manip = pipeline.createImageManip()
manip.setWaitForConfigInput(True)

manip_img = pipeline.createXLinkIn()
manip_img.setStreamName('manip_img')
manip_img.out.link(manip.inputImage)

manip_cfg = pipeline.createXLinkIn()
manip_cfg.setStreamName('manip_cfg')
manip_cfg.out.link(manip.inputConfig)

nn2 = pipeline.createNeuralNetwork()
nn2.setBlobPath(str((Path(__file__).parent / Path('models/text-recognition-0012.blob')).resolve().absolute()))
manip.out.link(nn2.input)

nn2_xout = pipeline.createXLinkOut()
nn2_xout.setStreamName("recognitions")
nn2.out.link(nn2_xout.input)

nn2_prev_xout = pipeline.createXLinkOut()
nn2_prev_xout.setStreamName("recognitions_pass")
nn2.passthrough.link(nn2_prev_xout.input)


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()


def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


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


class CTCCodec(object):
    """ Convert between text-label and text-index """
    def __init__(self, characters):
        # characters (str): set of the possible characters.
        dict_character = list(characters)

        self.dict = {}
        for i, char in enumerate(dict_character):
             self.dict[char] = i + 1

    
        self.characters = dict_character

    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds = preds.astype(np.float16)
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1, 0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        for l in preds_sizes:
            t = preds_index_reshape[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    if self.characters[t[i]] != '#':
                        char_list.append(self.characters[t[i]])
            text = ''.join(char_list)
            texts.append(text)

            index += l

        return texts

characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
codec = CTCCodec(characters)

frame = None

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)

running = True
detections = []
det_q = queue.Queue()


def det_thread(q_pass, dev):
    global detections, frame
    q_det = dev.getOutputQueue("detections")
    q_manip_cfg = dev.getInputQueue("manip_cfg")
    q_manip_img = dev.getInputQueue("manip_img")

    while running:
        try:
            in_rgb = q_pass.get()
            detections = q_det.get().detections
        except RuntimeError:
            continue

        q_manip_img.send(in_rgb)
        fps.tick_fps('det')

        for i, detection in enumerate(detections):
            cfg = dai.ImageManipConfig()
            cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            if i > 0:
                cfg.setReusePreviousImage(True)
            cfg.setResize(120, 32)
            q_manip_cfg.send(cfg)
            det_q.put(detection)


def rec_thread(dev):
    q_rec = dev.getOutputQueue("recognitions")
    q_pass = dev.getOutputQueue("recognitions_pass")

    while running:
        try:
            rec_data = np.array(q_rec.get().getFirstLayerFp16()).reshape(30, 1, 37)
            rec_frame = q_pass.get().getCvFrame()
            bbox = det_q.get(timeout=3)
        except RuntimeError:
            continue
        decoded_text = codec.decode(rec_data)[0]
        print(decoded_text)
        fps.tick_fps('rec')


with depthai.Device(pipeline) as device:
    device.startPipeline()

    q_pass = device.getOutputQueue("detections_pass")
    if args.camera:
        cam_out = device.getOutputQueue("preview", 1, True)
    else:
        nn_in = device.getInputQueue("nn_in")

    threads = [
        threading.Thread(target=det_thread, args=(q_pass, device, )),
        threading.Thread(target=rec_thread, args=(device, )),
    ]

    for t in threads:
        t.start()

    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        if args.video:
            return cap.read()
        else:
            return True, np.array(cam_out.get().getData()).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8)

    while should_run():
        read_correctly, frame = get_frame()

        if not read_correctly:
            break

        fps.next_iter()

        if not args.camera:
            tstamp = time.monotonic()
            data = to_planar(frame, (300, 300))
            img = dai.ImgFrame()
            img.setData(data)
            img.setTimestamp(tstamp)
            img.setWidth(300)
            img.setHeight(300)
            nn_in.send(img)

        if debug:
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            cv2.putText(frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.putText(frame, f"DET FPS:  {round(fps.tick_fps('det'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.putText(frame, f"REC FPS:  {round(fps.tick_fps('rec'), 1)}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.imshow('preview', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    running = False

for t in threads:
    t.join()
