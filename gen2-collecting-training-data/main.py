#!/usr/bin/env python3
import argparse
import queue
from pathlib import Path
from time import monotonic, strftime
from uuid import uuid4
from multiprocessing import Process, Queue
import cv2
import depthai as dai
import numpy as np


def check_range(min_val, max_val):
    def check_fn(value):
        ivalue = int(value)
        if min_val <= ivalue <= max_val:
            return ivalue
        else:
            raise argparse.ArgumentTypeError(
                "{} is an invalid int value, must be in range {}..{}".format(value, min_val, max_val)
            )
    return check_fn


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.3, type=float, help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--dirty', action='store_true', default=False, help="Allow the destination path not to be empty")
parser.add_argument('-nd', '--no-debug', dest="prod", action="store_true", default=False, help="Do not display debug output")
parser.add_argument('-m', '--time', type=float, default=float("inf"), help="Finish execution after X seconds")
parser.add_argument('-af', '--autofocus', type=str, default=None, help="Set AutoFocus mode of the RGB camera", choices=list(filter(lambda name: name[0].isupper(), vars(dai.CameraControl.AutoFocusMode))))
parser.add_argument('-mf', '--manualfocus', type=check_range(0, 255), help="Set manual focus of the RGB camera [0..255]")
parser.add_argument('-et', '--exposure-time', type=check_range(1, 33000), help="Set manual exposure time of the RGB camera [1..33000]")
parser.add_argument('-ei', '--exposure-iso', type=check_range(100, 1600), help="Set manual exposure ISO of the RGB camera [100..1600]")
parser.add_argument('-enc', '--encode', action="store_true", help="Encode the mono/color/depth output using H.264/H.265 encoding")
args = parser.parse_args()

exposure = [args.exposure_time, args.exposure_iso]
if any(exposure) and not all(exposure):
    raise RuntimeError("Both --exposure-time and --exposure-iso needs to be provided")


dest = Path(args.path).resolve().absolute()
dest_count = len(list(dest.glob('*')))
if dest.exists() and dest_count != 0 and not args.dirty:
    raise ValueError(f"Path {dest} contains {dest_count} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

pipeline = dai.Pipeline()

rgb = pipeline.createColorCamera()
rgb.setPreviewSize(300, 300)
rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setInterleaved(False)
rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(255)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
depth.setMedianFilter(median)
depth.setLeftRightCheck(False)
depth.setExtendedDisparity(False)
depth.setSubpixel(False)

left.out.link(depth.left)
right.out.link(depth.right)

controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
controlIn.out.link(rgb.inputControl)

# Create output
rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("color")
rgb.preview.link(rgbOut.input)
leftOut = pipeline.createXLinkOut()
leftOut.setStreamName("left")
left.out.link(leftOut.input)
rightOut = pipeline.createXLinkOut()
rightOut.setStreamName("right")
right.out.link(rightOut.input)
depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("disparity")
depth.disparity.link(depthOut.input)

if args.encode:
    # Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
    veL = pipeline.createVideoEncoder()
    veL.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
    left.out.link(veL.input)

    veC = pipeline.createVideoEncoder()
    veC.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
    rgb.video.link(veC.input)

    veR = pipeline.createVideoEncoder()
    veR.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
    right.out.link(veR.input)

    veDIn = pipeline.createXLinkIn()
    veDIn.setStreamName('ve_disparity_in')

    veD = pipeline.createVideoEncoder()
    veD.setDefaultProfilePreset(640, 400, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
    veDIn.out.link(veD.input)

    # Create outputs
    veLOut = pipeline.createXLinkOut()
    veLOut.setStreamName('ve_left')
    veL.bitstream.link(veLOut.input)

    veCOut = pipeline.createXLinkOut()
    veCOut.setStreamName('ve_color')
    veC.bitstream.link(veCOut.input)

    veROut = pipeline.createXLinkOut()
    veROut.setStreamName('ve_right')
    veR.bitstream.link(veROut.input)

    veDOut = pipeline.createXLinkOut()
    veDOut.setStreamName('ve_disparity')
    veD.bitstream.link(veDOut.input)


# https://stackoverflow.com/a/7859208/5494277
def step_norm(value):
    return round(value / args.threshold) * args.threshold


def seq(packet):
    return packet.getSequenceNum()


def tst(packet):
    return packet.getTimestamp().total_seconds()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


class PairingSystem:
    seq_streams = ["left", "right", "disparity"]
    ts_streams = ["color"]
    seq_ts_mapping_stream = "left"

    def __init__(self):
        self.ts_packets = {}
        self.seq_packets = {}
        self.last_paired_ts = None
        self.last_paired_seq = None

    def add_packets(self, packets, stream_name):
        if packets is None:
            return
        if stream_name in self.seq_streams:
            for packet in packets:
                seq_key = seq(packet)
                self.seq_packets[seq_key] = {
                    **self.seq_packets.get(seq_key, {}),
                    stream_name: packet
                }
        elif stream_name in self.ts_streams:
            for packet in packets:
                ts_key = step_norm(tst(packet))
                self.ts_packets[ts_key] = {
                    **self.ts_packets.get(ts_key, {}),
                    stream_name: packet
                }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.seq_streams):
                ts_key = step_norm(tst(self.seq_packets[key][self.seq_ts_mapping_stream]))
                if ts_key in self.ts_packets and has_keys(self.ts_packets[ts_key], self.ts_streams):
                    results.append({
                        **self.seq_packets[key],
                        **self.ts_packets[ts_key]
                    })
                    self.last_paired_seq = key
                    self.last_paired_ts = ts_key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]
        for key in list(self.ts_packets.keys()):
            if key <= self.last_paired_ts:
                del self.ts_packets[key]


extract_frame = {
    "left": lambda item: item.getCvFrame(),
    "right": lambda item: item.getCvFrame(),
    "color": lambda item: item.getCvFrame(),
    "disparity": lambda item: cv2.applyColorMap(item.getFrame(), cv2.COLORMAP_JET),
}

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


frame_q = Queue()


def store_frames(in_q):
    retries = 0
    while True:
        try:
            frames_dict = in_q.get(timeout=100)
        except queue.Empty:
            return
        if frames_dict is None:
            return
        frames_path = dest / Path(str(uuid4()))
        frames_path.mkdir(parents=False, exist_ok=False)
        for stream_name, item in frames_dict.items():
            cv2.imwrite(str(frames_path / Path(f"{stream_name}.png")), item)


def store_encodings(dev):
    encLeft = dev.getOutputQueue(name='ve_left', maxSize=30, blocking=True)
    encColor = dev.getOutputQueue(name='ve_color', maxSize=30, blocking=True)
    encRight = dev.getOutputQueue(name='ve_right', maxSize=30, blocking=True)
    encDisparity = device.getOutputQueue(name='ve_disparity', maxSize=30, blocking=True)

    timestamp_str = strftime("%Y%m%d-%H%M%S")
    fileLeft = open(f'data/left_{timestamp_str}.h264', 'wb')
    fileColor = open(f'data/color_{timestamp_str}.h265', 'wb')
    fileRight = open(f'data/right_{timestamp_str}.h264', 'wb')
    fileDisparity = open(f'data/disparity_{timestamp_str}.h264', 'wb')

    try:
        while True:
            if args.encode:
                while encLeft.has():
                    encLeft.get().getData().tofile(fileLeft)
                while encColor.has():
                    encColor.get().getData().tofile(fileColor)
                while encRight.has():
                    encRight.get().getData().tofile(fileRight)
                while encDisparity.has():
                    encDisparity.get().getData().tofile(fileDisparity)
    except RuntimeError:
        pass
    finally:
        fileLeft.close()
        fileColor.close()
        fileRight.close()
        print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
        cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
        print(cmd.format(fileLeft.name, str(Path(fileLeft.name).with_suffix('.mp4'))))
        print(cmd.format(fileRight.name, str(Path(fileRight.name).with_suffix('.mp4'))))
        print(cmd.format(fileColor.name, str(Path(fileColor.name).with_suffix('.mp4'))))


store_p = Process(target=store_frames, args=(frame_q, ))
store_p.start()
ps = PairingSystem()

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    qControl = device.getInputQueue('control')
    ve_disparity_in = device.getInputQueue(name='ve_disparity_in')
    enc_p = Process(target=store_encodings, args=(device,))
    enc_p.start()

    ctrl = dai.CameraControl()
    if args.autofocus:
        ctrl.setAutoFocusMode(getattr(dai.CameraControl.AutoFocusMode, args.autofocus))
    if args.manualfocus:
        ctrl.setManualFocus(args.manualfocus)
    if all(exposure):
        ctrl.setManualExposure(*exposure)

    qControl.send(ctrl)

    cfg = dai.ImageManipConfig()

    start_ts = monotonic()
    while True:
        for queueName in PairingSystem.seq_streams + PairingSystem.ts_streams:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if packets and queueName == "disparity":
                for packet in packets:
                    img = dai.ImgFrame()
                    img.setData(to_planar(extract_frame[queueName](packet), (640, 400)))
                    img.setTimestamp(monotonic())
                    img.setWidth(640)
                    img.setHeight(400)
                    ve_disparity_in.send(img)
            ps.add_packets(packets, queueName)

        pairs = ps.get_pairs()
        for pair in pairs:
            extracted_pair = {stream_name: extract_frame[stream_name](item) for stream_name, item in pair.items()}
            if not args.prod:
                for stream_name, item in extracted_pair.items():
                    cv2.imshow(stream_name, item)
            frame_q.put(extracted_pair)

        if not args.prod and cv2.waitKey(1) == ord('q'):
            break

        if monotonic() - start_ts > args.time:
            break

frame_q.put(None)
store_p.join()
enc_p.join()
