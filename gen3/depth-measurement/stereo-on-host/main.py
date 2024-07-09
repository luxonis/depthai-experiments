import cv2
import numpy as np
import depthai as dai

class StereoSGBM(dai.node.HostNode):
    def __init__(self):
        self.max_disparity = 96
        self.stereoProcessor = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=96,
            blockSize=5,
            P1=250, # 50
            P2=500, # 800
            disp12MaxDiff=5,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        super().__init__()

    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output, calibObj : dai.CalibrationHandler, monoCamera : dai.node.MonoCamera) -> "StereoSGBM":
        self.link_args(monoLeftOut, monoRightOut)
        self.sendProcessingToPipeline(True)

        self.baseline = calibObj.getBaselineDistance() * 10 # mm
        self.focal_length = self.count_focal_length(calibObj, monoCamera)
        self.H1, self.H2 = self.countH(calibObj, monoCamera)  # for left, right camera

        return self

    def process(self, monoLeft : dai.ImgFrame, monoRight : dai.ImgFrame) -> None:
        monoLeftFrame = monoLeft.getCvFrame()
        cv2.imshow("left", monoLeft.getCvFrame())

        monoRightFrame = monoRight.getCvFrame()
        cv2.imshow("right", monoRight.getCvFrame())

        self.create_disparity_map(monoLeftFrame, monoRightFrame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def countH(self, calibObj : dai.CalibrationHandler, monoCamera : dai.node.MonoCamera):
        width, height = self.get_width_height(monoCamera)

        M_left = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoLeftCameraId(), width, height))
        M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), width, height))

        R1 = np.array(calibObj.getStereoLeftRectificationRotation())
        R2 = np.array(calibObj.getStereoRightRectificationRotation())

        H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
        H_right = np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right))

        return H_left, H_right
    
    def count_focal_length(self, calibObj : dai.CalibrationHandler, monoCamera : dai.node.MonoCamera):
        width, height = self.get_width_height(monoCamera)

        M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), width, height))
        focalLength = M_right[0][0]
        return focalLength
    
    def get_width_height(self, monoCamera : dai.node.MonoCamera):
        width = monoCamera.getResolutionWidth()
        height = monoCamera.getResolutionHeight()
        return width, height

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

        #opencv skips disparity calculation for the first max_disparity pixels
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



device = dai.Device()
with dai.Pipeline(device) as pipeline:
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.CAM_B in cams and dai.CameraBoardSocket.CAM_C in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without left & right cameras! (Available cameras: {})".format(cams))
    
    calibObj = device.readCalibration() 

    host = pipeline.create(StereoSGBM).build(
        monoLeft.out,
        monoRight.out,
        calibObj,
        monoLeft
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
