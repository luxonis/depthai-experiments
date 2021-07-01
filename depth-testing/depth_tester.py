from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import depthai as dai
import numpy as np
import cv2
import argparse
import time


def pixel_coord_np(startX, startY, endX, endY):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    print(startX, startY, endX, endY)
    x = np.linspace(startX, endX - 1, endX - startX).astype(np.int32)
    y = np.linspace(startY, endY - 1, endY - startY).astype(np.int32)
    
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

parser = argparse.ArgumentParser()
parser.add_argument("-gt", "--groundTruth", type=float,required=True,
                            help="Set the actual disatance of the destination flat textured surface")
args = parser.parse_args()
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
    calibObj = device.readCalibration()
    # TODO(sachin): Change camera frame to right camera from rectified right.
    depthQueue = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

    M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), 1280, 720))
    R2 = np.array(calibObj.getStereoRightRectificationRotation())
    H_right_inv = np.linalg.inv(np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right)))
    inter_conv = np.matmul(np.linalg.inv(M_right), H_right_inv)

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
        
        if sbox is not None and ebox is not None:
            coord = pixel_coord_np(*sbox, *ebox)
            
            temp = depth_img.copy()
            temp = temp[sbox[1] : ebox[1], sbox[0] : ebox[0]]
            cam_coords = np.dot(inter_conv, coord) * temp.flatten() * 0.1 # [x, y, z]
            # points = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]])
            print(cam_coords.shape)
            subsampled_pixels = []
            subsampled_pixels_depth = []
            x_diff = ebox[0] - sbox[0]
            y_diff = ebox[1] - sbox[1]
            print("Start - {} and end - {}".format(sbox, ebox))
            for i in range(4):
                print(i)
                x_loc = sbox[0] + int(x_diff * (i/3)) 
                for j in range(4):
                    y_loc = sbox[1] + int(y_diff * (j/3))         
                    subsampled_pixels.append([x_loc, y_loc, 1])
                    subsampled_pixels_depth.append(depth_img[y_loc, x_loc])
                    print("Coordinate at {}, {} is {} & {} with depth of {}".format(i, j, x_loc, y_loc, depth_img[y_loc, x_loc]))
            subsampled_pixels_depth = np.array(subsampled_pixels_depth)
            subsampled_pixels = np.array(subsampled_pixels).transpose()
            
            sub_points3D = np.dot(inter_conv, subsampled_pixels) * subsampled_pixels_depth.flatten() # [x, y, z]
            print(sub_points3D.shape)
            # print(sub_points3D.transpose())
            
            points = Points(sub_points3D.transpose())
            print(points)
            plane = Plane.best_fit(points)
            points.plot_3d(ax, c='k', s=5, depthshade=True)
            plane.plot_3d(ax, alpha=0.2, lims_x=(0, 1000), lims_y=(0, 1000))

            fig.canvas.draw()
            fig.canvas.flush_events()
  
            time.sleep(0.1)

# plt.show()
""" points = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]])

plane = Plane.best_fit(points)

print(plane)
plot_3d(
    points.plotter(c='k', s=50, depthshade=False),
    plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
) """