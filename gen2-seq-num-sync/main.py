#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Create outputs
xout_left = pipeline.createXLinkOut()
xout_left.setStreamName('left')
cam_left.out.link(xout_left.input)
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName('right')
cam_right.out.link(xout_right.input)


def seq(packet):
    return packet.getSequenceNum()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


class PairingSystem:
    allowed_instances = [1, 2]  # Left (1) & Right (2)

    def __init__(self):
        self.seq_packets = {}
        self.last_paired_seq = None

    def add_packet(self, packet):
        if packet is not None and packet.getInstanceNum() in self.allowed_instances:
            seq_key = seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.getInstanceNum(): packet
            }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.allowed_instances):
                results.append(self.seq_packets[key])
                self.last_paired_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queue will be used to get the rgb frames from the output defined above
    q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    ps = PairingSystem()

    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        ps.add_packet(q_left.tryGet())
        ps.add_packet(q_right.tryGet())

        for synced in ps.get_pairs():
            raw_left = synced[1]
            raw_right = synced[2]

            frame_left = raw_left.getData().reshape((raw_left.getHeight(), raw_left.getWidth())).astype(np.uint8)
            frame_right = raw_right.getData().reshape((raw_right.getHeight(), raw_right.getWidth())).astype(np.uint8)

            cv2.imshow("left", frame_left)
            cv2.imshow("right", frame_right)

        if cv2.waitKey(1) == ord('q'):
            break
