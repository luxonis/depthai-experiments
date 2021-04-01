#!/usr/bin/env python3
import argparse
from pathlib import Path
from time import monotonic
from uuid import uuid4
from multiprocessing import Process
import cv2
import depthai as dai


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
parser.add_argument('-nd', '--no-debug', dest="prod", action='store_true', default=False, help="Do not display debug output")
parser.add_argument('-m', '--time', type=float, default=float("inf"), help="Finish execution after X seconds")
parser.add_argument('-af', '--autofocus', type=str, default=None, help="Set AutoFocus mode of the RGB camera", choices=list(filter(lambda name: name[0].isupper(), vars(dai.CameraControl.AutoFocusMode))))
parser.add_argument('-mf', '--manualfocus', type=check_range(0, 255), help="Set manual focus of the RGB camera [0..255]")
parser.add_argument('-et', '--exposure-time', type=check_range(1, 33000), help="Set manual exposure time of the RGB camera [1..33000]")
parser.add_argument('-ei', '--exposure-iso', type=check_range(100, 1600), help="Set manual exposure ISO of the RGB camera [100..1600]")
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

procs = []


def store_frames(frames_dict):
    global procs

    def store_frame(path, data, retries=0):
        try:
            cv2.imwrite(path, data)
        except OSError as ex:
            print("Failed to write frame to {}, error: {}".format(path, ex))
            if retries < 5:
                retries += 1
                print("Retrying to write frame to path {}... [{} retries left]".format(path, 5 - retries))
                return store_frame(path, data, retries)
            else:
                print("Frame at path {} will not be stored... [no retries left]")
                return

    frames_path = dest / Path(str(uuid4()))
    frames_path.mkdir(parents=False, exist_ok=False)
    new_procs = [
        Process(
            target=store_frame,
            args=(str(frames_path / Path(f"{stream_name}.png")), extract_frame[stream_name](item))
        )
        for stream_name, item in frames_dict.items()
    ]
    for proc in new_procs:
        proc.start()
    procs += new_procs


ps = PairingSystem()

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    qControl = device.getInputQueue('control')

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
            ps.add_packets(device.getOutputQueue(queueName).tryGetAll(), queueName)

        pairs = ps.get_pairs()
        for pair in pairs:
            if not args.prod:
                for stream_name, item in pair.items():
                    cv2.imshow(stream_name, extract_frame[stream_name](item))
            store_frames(pair)

        if not args.prod and cv2.waitKey(1) == ord('q'):
            break

        if monotonic() - start_ts > args.time:
            break

for proc in procs:
    proc.join()
