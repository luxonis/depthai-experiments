#!/usr/bin/env python3
import cv2
import depthai as dai
from depthai_sdk import FPSHandler
import time

FPS = 15

pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setInterleaved(True)
camRgb.setIspScale(1, 3)
camRgb.setPreviewSize(640, 360)
camRgb.setFps(FPS)

left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(FPS)

right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(FPS)

stereo = pipeline.createStereoDepth()
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

encLeft = pipeline.create(dai.node.VideoEncoder)
encLeft.setDefaultProfilePreset(left.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)
stereo.syncedLeft.link(encLeft.input)

encRight = pipeline.create(dai.node.VideoEncoder)
encRight.setDefaultProfilePreset(right.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)
stereo.syncedRight.link(encRight.input)

# Script node will sync high-res frames
script = pipeline.create(dai.node.Script)

stereo.disparity.link(script.inputs["disp_in"])
encLeft.bitstream.link(script.inputs["left_in"])
encRight.bitstream.link(script.inputs["right_in"])
camRgb.preview.link(script.inputs["rgb_in"])

script.setScript("""
FPS=15
import time
from datetime import timedelta
import math

MS_THRESHOL=math.ceil(500 / FPS)

def check_sync(queues, timestamp):
    matching_frames = []
    for name, list in queues.items():
        # node.warn(f"List {name}, len {str(len(list))}")
        for i, msg in enumerate(list):
            time_diff = abs(msg.getTimestamp() - timestamp)
            if time_diff <= timedelta(milliseconds=MS_THRESHOL):
                matching_frames.append(i)
                break

    if len(matching_frames) == len(queues):
        # We have all frames synced. Remove the excess ones
        i = 0
        for name, list in queues.items():
            queues[name] = queues[name][matching_frames[i]:]
            i+=1
        return True
    else:
        return False

names = ['disp', 'left', 'right', 'rgb']
frames = dict()
for name in names:
    frames[name] = []

while True:
    for name in names:
        f = node.io[name+"_in"].tryGet()
        if f is not None:
            frames[name].append(f)

            if check_sync(frames, f.getTimestamp()):
                # Frames synced!
                node.warn(f"Synced frame!")
                for name, list in frames.items():
                    syncedF = list.pop(0)
                    node.warn(f"{name}, ts: {str(syncedF.getTimestamp())}, seq {str(syncedF.getSequenceNum())}")
                    node.io[name+'_out'].send(syncedF)


    time.sleep(0.001)  # Avoid lazy looping
""")

script_out = ['disp', 'left', 'right']

for name in script_out:
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(name)
    script.outputs[name+'_out'].link(xout.input)

# Convert color stream RGB->NV12 and encode it
type_manip = pipeline.create(dai.node.ImageManip)
type_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.NV12)
script.outputs['rgb_out'].link(type_manip.inputImage)

encRgb = pipeline.create(dai.node.VideoEncoder)
encRgb.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)
type_manip.out.link(encRgb.input)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName('rgb')
encRgb.bitstream.link(xoutRgb.input)

with dai.Device(pipeline) as device:
    # Rgb should be the first - as we will first .get() that frame, as it will arrive the latest to the host
    names = ['rgb', 'disp', 'left', 'right']
    streams = {}
    for name in names:
        streams[name] = {
            'queue': device.getOutputQueue(name),
            'fps': FPSHandler(),  # Only for testing
        }

    while True:
        for name, stream in streams.items():
            img: dai.ImgFrame = stream['queue'].get()
            stream['fps'].next_iter()
            print(f"[{time.time()}] Stream {name} fps {stream['fps'].fps()}, timestamp: {img.getTimestamp()}, seq: {img.getSequenceNum()}")
