import cv2
import depthai as dai
import numpy as np
import math

class FilterWindow:
    def __init__(self, _lambda, _sigma) -> None:
        self._lambda = _lambda
        self._sigma = _sigma
        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        cv2.namedWindow("wlsFilter")

        cv2.createTrackbar("Lambda", "wlsFilter", 0, 255, self.on_trackbar_change_lambda)
        cv2.createTrackbar("Sigma", "wlsFilter", 0, 100, self.on_trackbar_change_sigma)
        cv2.setTrackbarPos("Lambda", "wlsFilter", 80)
        cv2.setTrackbarPos("Sigma", "wlsFilter", 15)

    def on_trackbar_change_lambda(self, value: int) -> None:
        self._lambda = value * 100
    def on_trackbar_change_sigma(self, value: int) -> None:
        self._sigma = value / float(10)

    def filter(self, disparity, right, depthScaleFactor) -> (np.ndarray, np.ndarray):
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
        self.wlsFilter.setLambda(self._lambda)
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
        self.wlsFilter.setSigmaColor(self._sigma)
        filteredDisp = self.wlsFilter.filter(disparity, right)

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
            # raw depth values
            depthFrame = (depthScaleFactor / filteredDisp).astype(np.uint16)

        return filteredDisp, depthFrame

class WLSFilter(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._filter_window = FilterWindow(_lambda=8000, _sigma=1.5)
        self._baseline = 75  # mm
        self._disp_levels = 96
        self._fov = 71.86

    def build(self, disparity: dai.Node.Output, rectified_right: dai.Node.Output,
              max_disparity: int) -> "WLSFilter":
        self.link_args(disparity, rectified_right)
        self.sendProcessingToPipeline(True)
        self._disp_multiplier = 255 / max_disparity
        return self

    def process(self, disparity_frame: dai.ImgFrame, right_frame: dai.ImgFrame) -> None:
        disparity_frame = disparity_frame.getFrame()
        right_frame = right_frame.getFrame()

        focal = disparity_frame.shape[1] / (2. * math.tan(math.radians(self._fov / 2)))
        depthScaleFactor = self._baseline * focal
        filteredDisp, depthFrame = self._filter_window.filter(disparity_frame, right_frame, depthScaleFactor)
        filteredDisp = (filteredDisp * self._disp_multiplier).astype(np.uint8)
        coloredDisp = cv2.applyColorMap(filteredDisp, cv2.COLORMAP_HOT)

        cv2.imshow("rectified right", right_frame)
        cv2.imshow("disparity", disparity_frame)
        cv2.imshow("wls raw depth", depthFrame)
        cv2.imshow("wlsFilter", filteredDisp)
        cv2.imshow("wls colored disp", coloredDisp)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
