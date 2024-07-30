#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse

parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')
parser.add_argument('-d', '--draw', default=False, action='store_true',
                    help='Draw on frames the sequence number and timestamp')
parser.add_argument('-v', '--verbose', default=0, action='count',
                    help='Verbose, -vv for more verbosity')
parser.add_argument('-t', '--dev_timestamp', default=False, action='store_true',
                    help='Get device timestamps, not synced to host. For debug')

args = parser.parse_args()


cam_list = ['left', 'rgb', 'right']

class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.window = ' + '.join(cam_list)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, (3*1280//2, 720//2))


    def build(self, rgb_out : dai.Node.Output, left_out : dai.Node.Output, right_out : dai.Node.Output) -> "Display":
        self.link_args(rgb_out, left_out, right_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, rgb_frame : dai.ImgFrame, left_frame : dai.ImgFrame, right_frame : dai.ImgFrame) -> None:
        frames = {
            "left" : left_frame.getCvFrame(), 
            "rgb" : rgb_frame.getCvFrame(), 
            "right" : right_frame.getCvFrame()
            }
        
        for c in cam_list:
            if c != 'rgb': 
                frames[c] = cv2.cvtColor(frames[c], cv2.COLOR_GRAY2BGR)

        frame_final = np.hstack(([frames[c] for c in cam_list]))
        cv2.imshow(self.window, frame_final)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


# Start defining a pipeline
with dai.Pipeline() as pipeline:

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setFps(args.fps)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setFps(args.fps)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(1080, 720)

    pipeline.create(Display).build(
        rgb_out=cam_rgb.preview,
        left_out=mono_left.out,
        right_out=mono_right.out
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
