import cv2
import numpy as np
import depthai as dai


class StereoSGBM:
    def __init__(self, baseline, H_right, H_left=np.identity(3, dtype=np.float32)):
        self.max_disparity = 96
        self.stereoProcessor = cv2.StereoSGBM_create(0, self.max_disparity, 21)
        self.baseline = baseline
        fov = 71.86
        self.focal_lenght = 1280 / (
                    2 * np.tan((fov * 3.142) / (2 * 180)))  # orig_frame_w / (2.f * std::tan(fov / 2 / 180.f * pi));
        # print(self.focal_lenght)
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

        # left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        # right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

        if is_rectify_enabled:
            left_img_rect, right_img_rect = self.rectification(left_img, right_img)  # Rectification using Homography
            # print("rectified")
        else:
            left_img_rect = left_img
            right_img_rect = right_img

        self.disparity = self.stereoProcessor.compute(left_img_rect, right_img_rect)
        cv2.filterSpeckles(self.disparity, 0, 60, self.max_disparity)

        _, self.disparity = cv2.threshold(self.disparity, 0, self.max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (self.disparity / 16.).astype(np.uint8)
        # frame = cv2.applyColorMap(disparity_scaled, cv2.COLORMAP_HOT)
        # frame[frame > 200] = 0
        # cv2.imshow("Scaled Disparity stream", frame)

        disparity_colour_mapped = cv2.applyColorMap(
            (disparity_scaled * (256. / self.max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT)
        cv2.imshow("Disparity", disparity_colour_mapped)
        cv2.imshow("rectified left", left_img_rect)
        cv2.imshow("rectified right", right_img_rect)

pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
xoutLeft = pipeline.createXLinkOut()
xoutRight = pipeline.createXLinkOut()

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
    raise RuntimeError("Unable to run this experiment on device without left & right cameras! (Available cameras: {})".format(cams))
calibObj = device.readCalibration()

M_left = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoLeftCameraId(), 1280, 720))
M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), 1280, 720))
R1 = np.array(calibObj.getStereoLeftRectificationRotation())
R2 = np.array(calibObj.getStereoRightRectificationRotation())

H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
H_right = np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right))

stereo_obj = StereoSGBM(7.5, H_right, H_left)

right = None
left = None

device.startPipeline(pipeline)
qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

while True:
    inLeft = qLeft.tryGet()
    inRight = qRight.tryGet()
    
    if inLeft is not None:
        cv2.imshow("left", inLeft.getCvFrame())
        left = inLeft.getCvFrame()

    if inRight is not None:
        cv2.imshow("right", inRight.getCvFrame())
        right = inRight.getCvFrame()

    if cv2.waitKey(1) == ord('q'):
        break
    
    if right is not None and left is not None:
        stereo_obj.create_disparity_map(left, right)

    if cv2.waitKey(1) == ord('q'):
        break
