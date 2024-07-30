#!/usr/bin/env python3

import cv2
import depthai as dai
import argparse
from datetime import timedelta
import math

def filter_internal_cameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


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


parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')

args = parser.parse_args()

cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.CAM_A,
    'left' : dai.CameraBoardSocket.CAM_B,
    'right': dai.CameraBoardSocket.CAM_C,
}
cam_instance = {
    'rgb'  : 0,
    'left' : 1,
    'right': 2,
}


# class Display(dai.node.HostNode):
#     def __init__(self) -> None:
#         super().__init__()


#     def build(self, rgb_out : dai.Node.Output, left_out : dai.Node.Output, right_out : dai.Node.Output, d_mxid) -> "Display":
#         self.d_mxid = d_mxid
#         self.link_args(rgb_out, left_out, right_out)
#         self.sendProcessingToPipeline(True)
#         return self
    

#     def process(self, rgb_frame : dai.ImgFrame, left_frame : dai.ImgFrame, right_frame : dai.ImgFrame) -> None:

#         cv2.imshow("left - " + self.d_mxid, left_frame.getCvFrame())
#         cv2.imshow("rgb - " + self.d_mxid, rgb_frame.getCvFrame())
#         cv2.imshow("right - " + self.d_mxid, right_frame.getCvFrame())

#         if cv2.waitKey(1) == ord('q'):
#             self.stopPipeline()


def create_pipeline(device : dai.Device, stereo : bool, queues : list):
    pipeline = dai.Pipeline(device)
    d_mxid = device.getMxId()

    if stereo:
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(args.fps)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setFps(args.fps)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        left_out = mono_left.out.createOutputQueue(maxSize=4, blocking=False)
        right_out = mono_right.out.createOutputQueue(maxSize=4, blocking=False)

        queues.append({
            'queue' : left_out,
            'msgs' : [],
            'name' : "left - " + d_mxid
        })
        queues.append({
            'queue' : right_out,
            'msgs' : [],
            'name' : "right - " + d_mxid
        })

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 3)

    rgb_out = cam_rgb.preview.createOutputQueue(maxSize=4, blocking=False)
    queues.append({
        'queue' : rgb_out,
        'msgs' : [],
        'name' : "rgb - " + d_mxid
    })

    return pipeline


device_infos = filter_internal_cameras(dai.Device.getAllAvailableDevices())

if len(device_infos) == 0: raise RuntimeError("No devices found!")
else: print("Found", len(device_infos), "devices")
queues = []
pipelines = []

for device_info in device_infos:
    device = dai.Device(device_info)

    stereo = 1 < len(device.getConnectedCameras())

    pipeline = create_pipeline(device, stereo, queues)
    pipelines.append(pipeline)
    pipeline.start()


while all(pipeline.isRunning() for pipeline in pipelines):
    
    for queue in queues:
        msg = queue['queue'].tryGet()
        if msg is not None:
            queue['msgs'].append(msg)
            if check_sync(queues, msg.getTimestamp()):
                for q in queues:
                    frame = q['msgs'].pop(0).getCvFrame()
                    cv2.imshow(q['name'], frame)


    if cv2.waitKey(1) == ord('q'):
        break
