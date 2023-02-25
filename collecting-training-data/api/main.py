#!/usr/bin/env python3
import argparse
from multiprocessing import Process, Queue
from pathlib import Path
from time import monotonic
from uuid import uuid4

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
parser.add_argument('-t', '--threshold', default=0.3, type=float,
                    help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--dirty', action='store_true', default=False,
                    help="Allow the destination path not to be empty")
parser.add_argument('-nd', '--no-debug', dest="prod", action='store_true', default=False,
                    help="Do not display debug output")
parser.add_argument('-m', '--time', type=float, default=float("inf"), help="Finish execution after X seconds")
parser.add_argument('-af', '--autofocus', type=str, default=None, help="Set AutoFocus mode of the RGB camera",
                    choices=list(filter(lambda name: name[0].isupper(), vars(dai.CameraControl.AutoFocusMode))))
parser.add_argument('-mf', '--manualfocus', type=check_range(0, 255),
                    help="Set manual focus of the RGB camera [0..255]")
parser.add_argument('-et', '--exposure-time', type=check_range(1, 33000),
                    help="Set manual exposure time of the RGB camera [1..33000]")
parser.add_argument('-ei', '--exposure-iso', type=check_range(100, 1600),
                    help="Set manual exposure ISO of the RGB camera [100..1600]")
args = parser.parse_args()

exposure = [args.exposure_time, args.exposure_iso]
if any(exposure) and not all(exposure):
    raise RuntimeError("Both --exposure-time and --exposure-iso needs to be provided")

dest = Path(args.path).resolve().absolute()
dest_count = len(list(dest.glob('*')))
if dest.exists() and dest_count != 0 and not args.dirty:
    raise ValueError(
        f"Path {dest} contains {dest_count} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)


def create_pipeline(depth_enabled=True):
    pipeline = dai.Pipeline()

    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setPreviewSize(300, 300)
    rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb.setInterleaved(False)
    rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')
    controlIn.out.link(rgb.inputControl)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    rgbOut.setStreamName("color")
    rgb.preview.link(rgbOut.input)

    if depth_enabled:
        left = pipeline.create(dai.node.MonoCamera)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)

        right = pipeline.create(dai.node.MonoCamera)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        depth = pipeline.create(dai.node.StereoDepth)
        depth.initialConfig.setConfidenceThreshold(255)
        median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
        depth.initialConfig.setMedianFilter(median)
        depth.setLeftRightCheck(False)
        depth.setExtendedDisparity(False)
        depth.setSubpixel(False)

        left.out.link(depth.left)
        right.out.link(depth.right)

        # Create output
        leftOut = pipeline.create(dai.node.XLinkOut)
        leftOut.setStreamName("left")
        left.out.link(leftOut.input)
        rightOut = pipeline.create(dai.node.XLinkOut)
        rightOut.setStreamName("right")
        right.out.link(rightOut.input)
        depthOut = pipeline.create(dai.node.XLinkOut)
        depthOut.setStreamName("disparity")
        depth.disparity.link(depthOut.input)

    return pipeline


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


def store_frames(in_q):
    while True:
        frames_dict = in_q.get()
        if frames_dict is None:
            return
        frames_path = dest / Path(str(uuid4()))
        frames_path.mkdir(parents=False, exist_ok=False)
        for stream_name, item in frames_dict.items():
            cv2.imwrite(str(frames_path / Path(f"{stream_name}.png")), item)


def main():
    frame_q = Queue(50)

    store_p = Process(target=store_frames, args=(frame_q,))
    store_p.start()

    try:
        # Pipeline defined, now the device is connected to
        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            ps = None
            if depth_enabled:
                ps = PairingSystem()
            else:
                PairingSystem.seq_streams = []
            device.startPipeline(create_pipeline(depth_enabled))
            qControl = device.getInputQueue('control')

            ctrl = dai.CameraControl()
            if args.autofocus:
                ctrl.setAutoFocusMode(getattr(dai.CameraControl.AutoFocusMode, args.autofocus))
            if args.manualfocus:
                ctrl.setManualFocus(args.manualfocus)
            if all(exposure):
                ctrl.setManualExposure(*exposure)

            qControl.send(ctrl)

            start_ts = monotonic()
            while True:
                for queueName in PairingSystem.seq_streams + PairingSystem.ts_streams:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if ps is not None:
                        ps.add_packets(packets, queueName)
                    elif queueName == "color":
                        for packet in packets:
                            frame_q.put({"color": extract_frame[queueName](packet)})
                    if queueName == "color" and len(packets) > 0 and not args.prod:
                        cv2.imshow("preview", packets[-1].getCvFrame())

                if ps is not None:
                    pairs = ps.get_pairs()
                    for pair in pairs:
                        extracted_pair = {stream_name: extract_frame[stream_name](item) for stream_name, item in
                                          pair.items()}
                        if not args.prod:
                            for stream_name, item in extracted_pair.items():
                                cv2.imshow(stream_name, item)
                        frame_q.put(extracted_pair)

                if not args.prod and cv2.waitKey(1) == ord('q'):
                    break

                if monotonic() - start_ts > args.time:
                    break
    finally:
        frame_q.put(None)
        store_p.join()


if __name__ == "__main__":
    main()
