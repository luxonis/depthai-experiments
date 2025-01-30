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
            P1=250,  # 50
            P2=500,  # 800
            disp12MaxDiff=5,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        super().__init__()

        self.mono_left = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.mono_right = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.disparity_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.rectified_left = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.rectified_right = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self,
        monoLeftOut: dai.Node.Output,
        monoRightOut: dai.Node.Output,
        calibObj: dai.CalibrationHandler,
        resolution: tuple[int, int],
    ) -> "StereoSGBM":
        self.link_args(monoLeftOut, monoRightOut)

        self.baseline = calibObj.getBaselineDistance() * 10  # mm
        self.focal_length = self.count_focal_length(calibObj, resolution)
        self.H1, self.H2 = self.count_h(calibObj, resolution)  # for left, right camera

        return self

    def process(self, monoLeft: dai.ImgFrame, monoRight: dai.ImgFrame) -> None:
        monoLeftFrame = monoLeft.getCvFrame()
        self.mono_left.send(
            self._create_img_frame(monoLeftFrame, dai.ImgFrame.Type.BGR888i)
        )

        monoRightFrame = monoRight.getCvFrame()
        self.mono_right.send(
            self._create_img_frame(monoRightFrame, dai.ImgFrame.Type.BGR888i)
        )

        self.create_disparity_map(monoLeftFrame, monoRightFrame)

    def count_h(self, calibObj: dai.CalibrationHandler, resolution: tuple[int, int]):
        width, height = resolution

        M_left = np.array(
            calibObj.getCameraIntrinsics(
                calibObj.getStereoLeftCameraId(), width, height
            )
        )
        M_right = np.array(
            calibObj.getCameraIntrinsics(
                calibObj.getStereoRightCameraId(), width, height
            )
        )

        R1 = np.array(calibObj.getStereoLeftRectificationRotation())
        R2 = np.array(calibObj.getStereoRightRectificationRotation())

        H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
        H_right = np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right))

        return H_left, H_right

    def count_focal_length(
        self, calibObj: dai.CalibrationHandler, resolution: tuple[int, int]
    ):
        width, height = resolution

        M_right = np.array(
            calibObj.getCameraIntrinsics(
                calibObj.getStereoRightCameraId(), width, height
            )
        )
        focalLength = M_right[0][0]
        return focalLength

    def rectification(self, left_img, right_img):
        # warp right image
        img_l = cv2.warpPerspective(
            left_img,
            self.H1,
            left_img.shape[:2][::-1],
            cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP,
        )

        img_r = cv2.warpPerspective(
            right_img,
            self.H2,
            right_img.shape[:2][::-1],
            cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP,
        )
        return img_l, img_r

    def create_disparity_map(self, left_img, right_img, is_rectify_enabled=True):
        if is_rectify_enabled:
            left_img_rect, right_img_rect = self.rectification(
                left_img, right_img
            )  # Rectification using Homography
        else:
            left_img_rect = left_img
            right_img_rect = right_img

        # opencv skips disparity calculation for the first max_disparity pixels
        pad_img_shape = [*left_img.shape]
        pad_img_shape[1] = self.max_disparity
        pad_img = np.zeros(shape=pad_img_shape, dtype=np.uint8)
        left_img_rect_pad = cv2.hconcat([pad_img, left_img_rect])
        right_img_rect_pad = cv2.hconcat([pad_img, right_img_rect])
        self.disparity = self.stereoProcessor.compute(
            left_img_rect_pad, right_img_rect_pad
        )
        self.disparity = self.disparity[
            0 : self.disparity.shape[0], self.max_disparity : self.disparity.shape[1]
        ]

        # scale back to integer disparities, opencv has 4 subpixel bits
        disparity_scaled = (self.disparity / 16.0).astype(np.uint8)

        disparity_colour_mapped = cv2.applyColorMap(
            (disparity_scaled * (256.0 / self.max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT,
        )

        self.disparity_out.send(
            self._create_img_frame(disparity_colour_mapped, dai.ImgFrame.Type.BGR888i)
        )
        self.rectified_left.send(
            self._create_img_frame(left_img_rect, dai.ImgFrame.Type.NV12)
        )
        self.rectified_right.send(
            self._create_img_frame(right_img_rect, dai.ImgFrame.Type.NV12)
        )

    def _create_img_frame(
        self, frame: np.ndarray, type: dai.ImgFrame.Type
    ) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(frame, type)
        return img_frame
