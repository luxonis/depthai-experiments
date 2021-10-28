#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
from datetime import timedelta
import contextlib
import math
import time
from pathlib import Path

# DepthAI Record library
from libraries.depthai_record import Record

_save_choices = ("color", "left", "right", "disparity", "depth") # TODO: depth/IMU/ToF...

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings", type=str, help="Path where to store the captured data")
parser.add_argument('-s', '--save', default=["color", "left", "right"], nargs="+", choices=_save_choices,
                    help="Choose which streams to save. Default: %(default)s")
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')
parser.add_argument('-ne', '--no_enc', default=False, action="store_true",
                    help='Disable encoding streams. Will consume more disk space')
parser.add_argument('-fc', '--frame_cnt', type=int, default=0,
                    help='Number of frames to record. Record until stopped by default.')
# TODO: make camera resolutions configrable
args = parser.parse_args()
save_path = Path.cwd() / args.path

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

def run_record():
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
            usb2_mode = False
            device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

            # Create recording object for this device
            recording = Record(str(save_path), device)
            # Set recording configuration
            # TODO: add support for specifying resolution, encoding quality
            recording.set_fps(args.fps)
            print(args.save)
            recording.set_save_streams(args.save)
            recording.set_encoding([] if args.no_enc else args.save)
            recording.start_recording()

            recordings.append(recording)

        queues = [q for recording in recordings for q in recording.queues]
        frame_counter = 0
        start_time = time.time()
        while True:
            try:
                for q in queues:
                    new_msg = q['q'].tryGet()
                    if new_msg is not None:
                        q['msgs'].append(new_msg)
                        if check_sync(queues, new_msg.getTimestamp()):
                            # Wait for Auto focus/exposure/white-balance
                            if time.time() - start_time < 1.3: continue

                            if args.frame_cnt == frame_counter: raise KeyboardInterrupt
                            frame_counter+=1

                            for recording in recordings:
                                frames = {}
                                for stream in recording.queues:
                                    frames[stream['name']] = stream['msgs'].pop(0).getCvFrame()
                                recording.frame_q.put(frames)
                if cv2.waitKey(1) == ord('q'):
                    break
            except KeyboardInterrupt:
                break

        for recording in recordings:
            recording.frame_q.put(None)
            recording.process.join() # Terminate the process
        print("All recordings have stopped. Exiting program")

if __name__ == '__main__':
    run_record()
