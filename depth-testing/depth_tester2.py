import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import cv2
import argparse
import depthai as dai

X = np.random.rand(100, 3)*10
Y = np.random.rand(100, 3)*5

sbox = None
ebox = None
color = (255, 0, 0)
parser = argparse.ArgumentParser()
parser.add_argument("-gt", "--groundTruth", type=float,required=True,
                            help="Set the actual disatance of the destination flat textured surface in cm")
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
    global sbox, ebox 
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = (x, y)
        # boxes.append(sbox)
    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = (x, y)
        # boxes.append(ebox)

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
x, y, z = [],[], []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(10,400)
plane = None

# stereo mode configs
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
fig.show()

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
                    if depth_img[y_loc, x_loc] == 0:
                        x_loc, y_loc = search_depth(x_loc, y_loc, depth_img)
                    subsampled_pixels_depth.append(depth_img[y_loc, x_loc])
                    print("Coordinate at {}, {} is {} & {} with depth of {}".format(i, j, x_loc, y_loc, depth_img[y_loc, x_loc]))
            subsampled_pixels_depth = np.array(subsampled_pixels_depth)
            subsampled_pixels = np.array(subsampled_pixels).transpose()

            sub_points3D = np.dot(inter_conv, subsampled_pixels) * subsampled_pixels_depth.flatten() * 0.1 # [x, y, z]
            print(sub_points3D.shape)
            print(sub_points3D.transpose())
            sub_points3D = sub_points3D.transpose()
            print(sub_points3D.shape)
            # points = Points(sub_points3D.transpose())
            sc._offsets3d = (sub_points3D[:,0], sub_points3D[:,1], sub_points3D[:,2])

            c, normal = fitPlaneLTSQ(sub_points3D)
            maxx = np.max(sub_points3D[:,0])
            maxy = np.max(sub_points3D[:,1])
            minx = np.min(sub_points3D[:,0])
            miny = np.min(sub_points3D[:,1])

            point = np.array([0.0, 0.0, c])
            d = -point.dot(normal)
            xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
            z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]
            print(xx, yy)
            if plane is None:
                plane = ax.plot_surface(xx, yy, z, alpha=0.2)
            else:
                plane_corners = np.vstack((xx.flatten(),yy.flatten(), z.flatten(), np.ones((1,4))))
                temp_col3 = plane_corners[:, 2].copy()
                plane_corners[:, 2] = plane_corners[:, 3]
                plane_corners[:, 3] = temp_col3
                # print(xx, yy)
                print(xx.flatten(), yy)
                print("Before")
                print(plane._vec)
                plane._vec = plane_corners
                print("After")
                print(plane._vec)

            print("Distance of subpixel from plane")
            print(sub_points3D[0].dot(normal) + d)


            plt.draw()
            print("Normal ~>")
            print(normal)
            print(d)
            print("Finding distance from all the points to the plane")
            print(cam_coords.shape)
            planeOffsetError = 0
            gtOffsetError = 0

            for roi_point in cam_coords.transpose():
                planeOffsetError += roi_point.dot(normal) + d
                gtOffsetError += roi_point.dot(gtPlane[0]) + gtPlane[1]
            print("Total Distance from the fitted plane = {}, Avg distance from the fitted plane = {}".format(planeOffsetError, planeOffsetError / cam_coords.shape[1]))
            print("Total Distance from the GT plane = {}, Avg distance from the GT plane = {}".format(gtOffsetError, gtOffsetError / cam_coords.shape[1]))
            
            plt.pause(0.1)
