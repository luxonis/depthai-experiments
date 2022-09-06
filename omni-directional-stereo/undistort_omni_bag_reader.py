import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
parser.add_argument("-bf" , "--bag_file", help="Input ROS bag.")
parser.add_argument("-c", "--cal", help="Calibration yaml file full path.")
parser.add_argument("-pr", "--perspectiveRotationPitch", type=float, required=False, default=0.0,
                        help="additional rotation to change the perspective angles.")
args = parser.parse_args()

calYaml = Path(args.cal)
rotAngle =  args.perspectiveRotationPitch
perspectiveRotationMatrix = Rotation.from_euler('y', rotAngle, degrees=True).as_matrix()
if not calYaml.exists():
    raise ValueError(
        'Calibration file not found: {}'.format(calYaml))

with open(calYaml) as file:
    camChainDict = yaml.load(file, Loader=yaml.FullLoader)


sbox = None
ebox = None
localSbox = None
localEbox = None
isBoxCompleted = False

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


cv2.namedWindow('disparityColor')
cv2.setMouseCallback('disparityColor', on_mouse)

maxDisparity = 190
blockSize = 5
K = 32
LRthreshold = 2
stereoProcessor = cv2.StereoSGBM_create(
    minDisparity=1,
    numDisparities=maxDisparity,
    blockSize=blockSize,
    P1=2 * (blockSize ** 2),
    P2=K * (blockSize ** 2),
    disp12MaxDiff=LRthreshold,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

bag = rosbag.Bag(args.bag_file, "r")
bridge = CvBridge()

count  = 0
topic_names = ["/stereo_inertial_publisher/left/image_raw", "/stereo_inertial_publisher/right/image_raw"]
leftImg = None
rightImg = None


intrinsic = camChainDict['cam0']['intrinsics']
xiLeft = np.array([[intrinsic[0]]], float)
kLeft = np.array([[intrinsic[1], 0, intrinsic[3]], 
                        [0, intrinsic[2], intrinsic[4]], 
                        [0, 0, 1.0]], float)

print(camChainDict['cam0']['distortion_coeffs'])
dLeft = np.array(camChainDict['cam0']['distortion_coeffs'], float)

intrinsic = camChainDict['cam1']['intrinsics']
xiRight = np.array([[intrinsic[0]]], float)
kRight = np.array([[intrinsic[1], 0, intrinsic[3]], 
                    [0, intrinsic[2], intrinsic[4]], 
                    [0, 0, 1.0]], float)
dRight = np.array(camChainDict['cam1']['distortion_coeffs'], float)

transformation = np.array(camChainDict['cam1']['T_cn_cnm1'], float)
R = transformation[:3, :3]
t = transformation[:3, 3]
# print(t)

R1, R2 = cv2.omnidir.stereoRectify(R, t)
R1 = np.matmul(R1, perspectiveRotationMatrix)
R2 = np.matmul(R2, perspectiveRotationMatrix)

color = (0, 0, 255)
leftCount = 0
rightCount = 0

for topic, msg, rec_time in bag.read_messages(topics=topic_names):
    cv_img =  bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    if "left" in topic:
        leftImg = cv_img
        leftCount += 1
    elif "right" in topic:
        rightImg = cv_img
        rightCount += 1
    # print(topic)

    if leftCount != rightCount:
        continue


    undistortedPerspectiveLeft = cv2.omnidir.undistortImage(leftImg,   kLeft, dLeft, xiLeft, cv2.omnidir.RECTIFY_PERSPECTIVE, R = R1)
    undistortedPerspectiveRight = cv2.omnidir.undistortImage(rightImg, kRight, dRight, xiRight, cv2.omnidir.RECTIFY_PERSPECTIVE, R = R2)

    subpixelBits = 16.
    disparity = stereoProcessor.compute(undistortedPerspectiveLeft, undistortedPerspectiveRight)

    disparity = (disparity / subpixelBits)
    disparityDisp = disparity.astype(np.uint8)

    im_color = cv2.applyColorMap(disparityDisp, cv2.COLORMAP_JET)
    im_color = cv2.rectangle(im_color, sbox, ebox, color, 2)

    cv2.imshow("undistortedPerspectiveLeft", undistortedPerspectiveLeft)
    cv2.imshow("undistortedPerspectiveRight", undistortedPerspectiveRight)
    cv2.imshow("disparity", disparityDisp)
    cv2.imshow("disparityColor", im_color)
    if isBoxCompleted:
        sbox = (min(localSbox[0], localEbox[0]), min(localSbox[1], localEbox[1]))
        ebox = (max(localSbox[0], localEbox[0]), max(localSbox[1], localEbox[1]))

    if sbox is not None and ebox is not None:
        if (ebox[0] - sbox[0]) < 50  and (ebox[1] - sbox[1]) < 50:
            print("Requires the ROI to be of shapa greater than 50x50")
        else:
            croppedArea = disparity[sbox[1] : ebox[1], sbox[0] : ebox[0]]
            dispMedian = np.median(croppedArea)
            # print(f'cropped area sioze {croppedArea.shape}')
            print(f'dispMedian is {dispMedian} and t[0] is {t[0]}')

            depth = np.abs(t[0]) * kLeft[0,0] / dispMedian 
            print(f'depth is {depth} with T being {t[0]} using left Intrinsics')
            t_abs = np.linalg.norm(t)
            depth = t_abs * kLeft[0,0] / dispMedian 
            print(f'depth is {depth} with T being {t_abs} using abs')


    key = cv2.waitKey(1)
    if key == ord('q'):
        exit(0)


