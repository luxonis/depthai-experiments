import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import cv2
import argparse
import depthai as dai
import math

X = np.random.rand(100, 3)*10
Y = np.random.rand(100, 3)*5

sbox = None
ebox = None
localSbox = None
localEbox = None
isBoxCompleted = False

color = (255, 0, 0)
parser = argparse.ArgumentParser()
parser.add_argument("-gt", "--groundTruth", type=float,required=True,
                            help="Set the actual disatance of the destination flat textured surface in mtrs")
args = parser.parse_args()
gtNormal = np.array([0, 0, 1], dtype=np.float32)
gtD = - args.groundTruth
gtPlane = (gtNormal, gtD)

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

def search_depth(x, y, depth):
    up_x, up_y, status = search_depth_pos(x, y, depth)
    if not status:
        up_x, up_y, status = search_depth_neg(x, y, depth)
    return up_x, up_y
    
def search_depth_pos(x, y, depth):
    def_x = x
    def_y = y
    while depth[y, x] == 0:
        if depth[y + 1, x] != 0:
            y += 1
        elif depth[y, x + 1] != 0:
            x += 1
        else:
            y += 1
            x += 1
        if abs(def_x - x) > 10 or abs(def_y - y) > 10:
            return x, y, False
    return x, y, True

def search_depth_neg(x, y, depth):
    def_x = x
    def_y = y
    while depth[y, x] == 0:
        if depth[y - 1, x] != 0:
            y -= 1
        elif depth[y, x - 1] != 0:
            x -= 1
        else:
            y -= 1
            x -= 1
        if abs(def_x - x) > 10 or abs(def_y - y) > 10:
            return x, y, False
    return x, y, True


def on_mouse(event, x, y, flags, params):
    global localSbox, localEbox, isBoxCompleted 
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: '+str(x)+', '+str(y))
        localSbox = (x, y)
        isBoxCompleted = False
        # boxes.append(sbox)
    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        localEbox = (x, y)
        isBoxCompleted = True

def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)

# Plot Setup
plt.ion()
x, y, z = [], [], []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1,5)
ax.set_xlim(-0.3,0.3)
ax.set_ylim(-0.3,0.3)

plotFitPlane = None

# stereo mode configs
lrcheck = False
extended = False
subpixel = False

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
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setConfidenceThreshold(200)
stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
# stereo.setInputResolution(1280, 720)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

cv2.namedWindow('depth')
fig.show()

with dai.Device(pipeline) as device:
    calibObj = device.readCalibration()
    # TODO(sachin): Change camera frame to right camera from rectified right.
    depthQueue = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

    M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), 1280, 720))
    R2 = np.array(calibObj.getStereoRightRectificationRotation())
    H_right_inv = np.linalg.inv(np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right)))
    inter_conv = np.matmul(np.linalg.inv(M_right), H_right_inv)
    cv2.setMouseCallback('depth', on_mouse)

    while True:
        depth = depthQueue.get()
        depth_img = depth.getCvFrame()
        frame = depth_img.copy()

        cv2.rectangle(frame, sbox, ebox, color, 2)
        # if sbox is not None and ebox is not None:
        #     cv2.rectangle(frame, sbox, ebox, color, 2)
        # cv2.imshow("depth", frame.astype(np.uint16))
        depthFrameColor = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)        
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        if isBoxCompleted:
            sbox = (min(localSbox[0], localEbox[0]), min(localSbox[1], localEbox[1]))
            ebox = (max(localSbox[0], localEbox[0]), max(localSbox[1], localEbox[1]))
        else:
            continue

        if (ebox[0] - sbox[0]) < 50  and (ebox[1] - sbox[1]) < 50:
            print("Requires the ROI to be of shapa greater than 50x50")

        if sbox is not None and ebox is not None:
            print("Start - {} and end - {}".format(sbox, ebox))
            
            coord = pixel_coord_np(*sbox, *ebox)
            roi = depth_img.copy()
            roi = roi[sbox[1] : ebox[1], sbox[0] : ebox[0]]
            
            camCoords = np.dot(inter_conv, coord) * roi.flatten() / 1000.0 # [x, y, z]
            # Removing Zeros from coordinates
            validCamCoords = np.delete(camCoords, np.where(camCoords[2, :] == 0.0), axis=1)
            # Removing outliers from Z coordinates. top and bottoom 0.5 percentile of valid depth
            validCamCoords = np.delete(validCamCoords, np.where(validCamCoords[2, :] <= np.percentile(validCamCoords[2, :], 0.5)), axis=1)
            validCamCoords = np.delete(validCamCoords, np.where(validCamCoords[2, :] >= np.percentile(validCamCoords[2, :], 99.5)), axis=1)

            # Subsampling 4x4 grid points in the selected ROI
            subsampledPixels = []
            subsampledPixelsDepth = []
            x_diff = ebox[0] - sbox[0]
            y_diff = ebox[1] - sbox[1]
            for i in range(4):
                x_loc = sbox[0] + int(x_diff * (i/3)) 
                for j in range(4):
                    y_loc = sbox[1] + int(y_diff * (j/3))
                    subsampledPixels.append([x_loc, y_loc, 1])
                    if depth_img[y_loc, x_loc] == 0:
                        x_loc, y_loc = search_depth(x_loc, y_loc, depth_img)
                    subsampledPixelsDepth.append(depth_img[y_loc, x_loc])
                    # print("Coordinate at {}, {} is {} & {} with depth of {}".format(i, j, x_loc, y_loc, depth_img[y_loc, x_loc]))

            subsampledPixelsDepth = np.array(subsampledPixelsDepth)
            subsampledPixels = np.array(subsampledPixels).transpose()

            sub_points3D = np.dot(inter_conv, subsampledPixels) * subsampledPixelsDepth.flatten() / 1000.0 # [x, y, z]
            sub_points3D = sub_points3D.transpose()
            sc._offsets3d = (sub_points3D[:, 0], sub_points3D[:, 1], sub_points3D[:, 2])
            
            # Fitting the plane to the subsampled RPOI points
            c, normal = fitPlaneLTSQ(sub_points3D)
            maxx = np.max(sub_points3D[:,0])
            maxy = np.max(sub_points3D[:,1])
            minx = np.min(sub_points3D[:,0])
            miny = np.min(sub_points3D[:,1])

            d = -np.array([0.0, 0.0, c]).dot(normal)
            fitPlane = (normal, d)
            xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
            z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]

            # Plotting the fit plane
            if plotFitPlane is None:
                plotFitPlane = ax.plot_surface(xx, yy, z, alpha=0.2)
            else:
                fitPlaneCorners = np.vstack((xx.flatten(),yy.flatten(), z.flatten(), np.ones((1,4))))
                temp_col3 = fitPlaneCorners[:, 2].copy()
                fitPlaneCorners[:, 2] = fitPlaneCorners[:, 3]
                fitPlaneCorners[:, 3] = temp_col3
                plotFitPlane._vec = fitPlaneCorners


            print("Distance of subpixel from plane")
            print(sub_points3D[0].dot(normal) + d)

            plt.draw()

            # Calculating the error
            planeOffsetError = 0
            gtOffsetError = 0
            planeRmsOffsetError = 0
            gtRmsOffsetError = 0

            for roiPoint in validCamCoords.transpose():
                fitPlaneDist = roiPoint.dot(normal) + d
                gtPlaneDist = roiPoint.dot(gtPlane[0]) + gtPlane[1]
                planeOffsetError += fitPlaneDist
                gtOffsetError += gtPlaneDist
                planeRmsOffsetError += fitPlaneDist**2
                gtRmsOffsetError += gtPlaneDist**2

            print("Plane Fit MSE: Avg depth pixels deviation from the calculated Plane Fit = {}".format(planeOffsetError / validCamCoords.shape[1]))
            print("GT Plane MSE: Avg depth pixels deviation from the GT plane = {}".format(gtOffsetError / validCamCoords.shape[1]))
            print("Plane Fit RMSE: Avg distance of points in ROI from the fitted plane = {}".format(math.sqrt(planeRmsOffsetError / validCamCoords.shape[1])))
            print("GT Plane RMSE: Avg distance of points in ROI from the GT plane = {}".format(math.sqrt(gtRmsOffsetError / validCamCoords.shape[1])))
            
            totalPixels = (ebox[0] - sbox[0]) * (ebox[1] - sbox[1])
            flatRoi = roi.flatten()
            sampledPixels = np.delete(flatRoi, np.where(flatRoi == 0))
            fill_rate = 100 * sampledPixels.shape[0] / totalPixels
            print('fill rate is {}'.format(fill_rate))

            plt.pause(0.1)
