#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
from datetime import timedelta
import contextlib
import math
import time
# DepthAI Record library
from libraries.depthai_record import Record

_save_choices = ("color", "mono") # TODO: depth/IMU/ToF...

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings", type=str, help="Path where to store the captured data")
parser.add_argument('-s', '--save', default=["color", "mono"], nargs="+", choices=_save_choices,
                    help="Choose which streams to save. Default: %(default)s")
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')
# TODO: make camera resolutions configrable

args = parser.parse_args()

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

# Host side timestamp frame sync across multiple devices
def check_sync(queues, timestamp):
    matching_frames = []
    for q in queues:
        for i, msg in enumerate(q['msgs']):
            time_diff = abs(msg.getTimestamp() - timestamp)
            # So below 17ms @ 30 FPS => frames are in sync
            if time_diff <= timedelta(milliseconds=math.ceil(500 / args.fps)):
                matching_frames.append(i)
                break

    if len(matching_frames) == len(queues):
        # We have all frames synced. Remove the excess ones
        for i, q in enumerate(queues):
            q['msgs'] = q['msgs'][matching_frames[i]:]
        return True
    else:
        return False

# Record from all available devices
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")

    recordings = []
    # TODO: allow users to specify which available devices should record
    for device_info in device_infos:
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = True
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

        # Create recording object for this device
        recording = Record(args.path, device)
        # Set recording configuration
        # TODO: add support for specifying resolution, encoding quality
        recording.set_fps(args.fps)
        recording.set_save_streams(args.save)
        recording.start_recording()

        recordings.append(recording)

    queues = [q for recording in recordings for q in recording.queues]
    while True:
        for q in queues:
            new_msg = q['q'].tryGet()
            if new_msg is not None:
                q['msgs'].append(new_msg)
                if check_sync(queues, new_msg.getTimestamp()):
                    # print('frames synced')
                    for recording in recordings:
                        frames = {}
                        for stream in recording.queues:
                            frames[stream['name']] = stream['msgs'].pop(0).getCvFrame()
                            # cv2.imshow(f"{stream['name']} - {device['mx']}", cv2.imdecode(frames[stream['name']], cv2.IMREAD_UNCHANGED))
                        # print('For mx', device['mx'], 'frames')
                        # print('frames', frames)
                        recording.frame_q.put(frames)
        if cv2.waitKey(1) == ord('q'):
            break

    for recording in recordings:
        recording.frame_q.put(None)
        time.sleep(0.01) # Wait 10ms for process to close all video files
        recording.process.join() # Terminate the process

