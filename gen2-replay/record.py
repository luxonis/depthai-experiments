#!/usr/bin/env python3
import argparse
from pathlib import Path
from time import monotonic
from multiprocessing import Process, Queue
import cv2
import depthai as dai
import contextlib


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
# parser.add_argument('-nd', '--no-debug', dest="prod", action='store_true', default=False, help="Do not display debug output")
parser.add_argument('-m', '--time', type=float, default=float("inf"), help="Finish execution after X seconds")

parser.add_argument('-nd', '--no-depth', action='store_true', default=False, help="Do not save depth map. If set, mono frames will be saved")
parser.add_argument('-mono', '--mono', action='store_true', default=False, help="Save mono frames")
parser.add_argument('-e', '--encode', action='store_true', default=False, help="Encode mono frames into jpeg. If set, it will enable -mono as well")

args = parser.parse_args()

SAVE_MONO = args.encode or args.mono or args.no_depth

dest = Path(args.path).resolve().absolute()
dest_count = len(list(dest.glob('*')))
if dest.exists() and dest_count != 0 and not args.dirty:
    raise ValueError(f"Path {dest} contains {dest_count} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

pipeline = dai.Pipeline()

rgb = pipeline.createColorCamera()

rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setInterleaved(False)
rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create output for the rgb
rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("color")

rgb_encoder = pipeline.createVideoEncoder()
rgb_encoder.setDefaultProfilePreset(rgb.getVideoSize(), rgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
rgb_encoder.setLossless(True)
rgb.video.link(rgb_encoder.input)
rgb_encoder.bitstream.link(rgbOut.input)

# Create mono cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(240)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
stereo.setMedianFilter(median)
stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

left.out.link(stereo.left)
right.out.link(stereo.right)

if not args.no_depth:
    depthOut = pipeline.createXLinkOut()
    depthOut.setStreamName("depth")
    stereo.depth.link(depthOut.input)

controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
controlIn.out.link(rgb.inputControl)

# Create output
if SAVE_MONO:
    leftOut = pipeline.createXLinkOut()
    leftOut.setStreamName("left")
    rightOut = pipeline.createXLinkOut()
    rightOut.setStreamName("right")
    if args.encode:
        left_encoder = pipeline.createVideoEncoder()
        left_encoder.setDefaultProfilePreset(left.getResolutionSize(), left.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
        left_encoder.setLossless(True)
        stereo.rectifiedLeft.link(left_encoder.input)
        left_encoder.bitstream.link(leftOut.input)

        right_encoder = pipeline.createVideoEncoder()
        right_encoder.setDefaultProfilePreset(right.getResolutionSize(), right.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
        right_encoder.setLossless(True)
        stereo.rectifiedRight.link(right_encoder.input)
        right_encoder.bitstream.link(rightOut.input)
    else:
        stereo.rectifiedLeft.link(leftOut.input)
        stereo.rectifiedRight.link(rightOut.input)

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
    def __init__(self, save_mono, save_depth):
        self.ts_packets = {}
        self.seq_packets = {}
        self.last_paired_ts = None
        self.last_paired_seq = None

        self.seq_streams = []
        if save_mono:
            self.seq_streams.append("left")
            self.seq_streams.append("right")
        if save_depth:
            self.seq_streams.append("depth")
        print(self.seq_streams)
        self.ts_streams = ["color"]
        self.seq_ts_mapping_stream = self.seq_streams[0]

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

class Record:
    def __init__(self, dest, device, mxId):
        self.ps = PairingSystem(SAVE_MONO, not args.no_depth)
        self.folder_num = 0

        self.dest = dest
        if mxId is not None: self.dest = self.dest / mxId

        self.frame_q = Queue()
        self.device = device

        self.store_p = Process(target=self.store_frames, args=(self.frame_q, ))
        self.store_p.start()

    def store_frames(self, in_q):
        def save_png(frames_path, name, item):
            cv2.imwrite(str(frames_path / f"{name}.png"), item)
        def save_jpeg(frames_path, name, item):
            with open(str(frames_path / f"{name}.jpeg"), "wb") as f:
                f.write(bytearray(item))
        def save_depth(frames_path, name, item):
            with open(str(frames_path / f"{name}"), "wb") as f:
                f.write(bytearray(item))
        while True:
            frames_dict = in_q.get()
            if frames_dict is None:
                return
            frames_path = self.dest / str(self.folder_num)
            frames_path.mkdir(parents=True, exist_ok=False)
            self.folder_num += 1
            for stream_name, item in frames_dict.items():
                if stream_name == "depth": save_depth(frames_path, stream_name, item)
                elif stream_name == "color": save_jpeg(frames_path, stream_name, item)
                elif args.encode: save_jpeg(frames_path, stream_name, item)
                else: save_png(frames_path, stream_name, item)

    def get_queues(self):
        for queueName in self.ps.seq_streams + self.ps.ts_streams:
            self.ps.add_packets(self.device.getOutputQueue(queueName).tryGetAll(), queueName)

        pairs = self.ps.get_pairs()
        for pair in pairs:
            obj = { "color": pair["color"].getData() }
            if SAVE_MONO:
                if args.encode:
                    obj["left"] = pair["left"].getData()
                    obj["right"] = pair["right"].getData()
                else:
                    obj["left"] = pair["left"].getCvFrame()
                    obj["right"] = pair["right"].getCvFrame()
            if not args.no_depth:
                obj["depth"] = pair["depth"].getFrame()

            self.frame_q.put(obj)
    def close(self):
        self.frame_q.put(None)
        self.store_p.join()

# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    recording_devices = []
    for device_info in dai.Device.getAllAvailableDevices():
        device = stack.enter_context(dai.Device(pipeline, device_info))
        print(f"Connecting to MX", device_info.getMxId())
        recording_devices.append(Record(dest=dest, device=device, mxId=device_info.getMxId()))
        device.startPipeline()

    start_ts = monotonic()
    while True:
        for recording_device in recording_devices:
            recording_device.get_queues()
            if cv2.waitKey(1) == ord('q'):
                break
            if monotonic() - start_ts > args.time:
                break
    # Close all devices
    for recording_device in recording_devices:
        recording_device.close()

