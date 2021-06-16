from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
import depthai as dai
import numpy as np
import cv2
import argparse


sbox = None
ebox = None
color = (255, 0, 0)
def on_mouse(event, x, y, flags, params):
    global sbox, ebox 
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = (x, y)
        # boxes.append(sbox)
    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = (x, y)
        # boxes.append(ebox)
        
lrcheck = True
extended = False
subpixel = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutDepth = pipeline.createXLinkOut()

# XLinkOut
xoutDepth.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setConfidenceThreshold(200)
stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
# stereo.setInputResolution(1280, 720)
stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

cv2.namedWindow('depth')
with dai.Device(pipeline) as device:
    # TODO(sachin): Change camera frame to right camera from rectified right.
    depthQueue = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

    while True:
        depth = depthQueue.get()
        depth_img = depth.getCvFrame()
        frame = depth_img.copy()

        cv2.setMouseCallback('depth', on_mouse)
        if sbox is not None and ebox is not None:
            cv2.rectangle(frame, sbox, ebox, color, 2)
        # cv2.imshow("depth", frame.astype(np.uint16))
        depthFrameColor = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)        
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        


points = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]])

plane = Plane.best_fit(points)

print(plane)
plot_3d(
    points.plotter(c='k', s=50, depthshade=False),
    plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
)


