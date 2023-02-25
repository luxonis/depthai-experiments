import cv2
import depthai as dai
import numpy as np


class StereoSGBM:
    def __init__(self, baseline, focalLength, H_right, H_left=np.identity(3, dtype=np.float32)):
        self.max_disparity = 96
        self.stereoProcessor = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=96,
            blockSize=5,
            P1=250,  # 50
            P2=500,  # 800
            disp12MaxDiff=5,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.baseline = baseline
        self.focal_length = focalLength
        self.H1 = H_left  # for left camera
        self.H2 = H_right  # for right camera

    def rectification(self, left_img, right_img):
        # warp right image
        img_l = cv2.warpPerspective(left_img, self.H1, left_img.shape[::-1],
                                    cv2.INTER_CUBIC +
                                    cv2.WARP_FILL_OUTLIERS +
                                    cv2.WARP_INVERSE_MAP)

        img_r = cv2.warpPerspective(right_img, self.H2, right_img.shape[::-1],
                                    cv2.INTER_CUBIC +
                                    cv2.WARP_FILL_OUTLIERS +
                                    cv2.WARP_INVERSE_MAP)
        return img_l, img_r

    def create_disparity_map(self, left_img, right_img, is_rectify_enabled=True):

        if is_rectify_enabled:
            left_img_rect, right_img_rect = self.rectification(left_img, right_img)  # Rectification using Homography
        else:
            left_img_rect = left_img
            right_img_rect = right_img

        # opencv skips disparity calculation for the first max_disparity pixels
        padImg = np.zeros(shape=[left_img.shape[0], self.max_disparity], dtype=np.uint8)
        left_img_rect_pad = cv2.hconcat([padImg, left_img_rect])
        right_img_rect_pad = cv2.hconcat([padImg, right_img_rect])
        self.disparity = self.stereoProcessor.compute(left_img_rect_pad, right_img_rect_pad)
        self.disparity = self.disparity[0:self.disparity.shape[0], self.max_disparity:self.disparity.shape[1]]

        # scale back to integer disparities, opencv has 4 subpixel bits
        disparity_scaled = (self.disparity / 16.).astype(np.uint8)

        disparity_colour_mapped = cv2.applyColorMap(
            (disparity_scaled * (256. / self.max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT)
        cv2.imshow("Disparity", disparity_colour_mapped)
        cv2.imshow("rectified left", left_img_rect)
        cv2.imshow("rectified right", right_img_rect)


pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

device = dai.Device()
cams = device.getConnectedCameras()
depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
if not depth_enabled:
    raise RuntimeError(
        "Unable to run this experiment on device without left & right cameras! (Available cameras: {})".format(cams))
calibObj = device.readCalibration()

width = monoLeft.getResolutionWidth()
height = monoLeft.getResolutionHeight()

M_left = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoLeftCameraId(), width, height))
M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), width, height))
R1 = np.array(calibObj.getStereoLeftRectificationRotation())
R2 = np.array(calibObj.getStereoRightRectificationRotation())

H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
H_right = np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right))

baseline = calibObj.getBaselineDistance() * 10  # mm
focalLength = M_right[0][0]
stereo_obj = StereoSGBM(baseline, focalLength, H_right, H_left)

with device:
    device.startPipeline(pipeline)
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    while True:
        inLeft = qLeft.get()
        inRight = qRight.get()

        cv2.imshow("left", inLeft.getCvFrame())
        left = inLeft.getCvFrame()

        cv2.imshow("right", inRight.getCvFrame())
        right = inRight.getCvFrame()

        stereo_obj.create_disparity_map(left, right)

        if cv2.waitKey(1) == ord('q'):
            break
