import depthai as dai
import numpy as np


class HostSpatialsCalc:
    # We need device object to get calibration data
    def __init__(
        self,
        calib_data: dai.CalibrationHandler,
        depth_alignment_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
    ):
        self.calibData = calib_data
        self.depth_alignment_socket = depth_alignment_socket

        # Values
        self.DELTA = 5
        self.THRESH_LOW = 200  # 20cm
        self.THRESH_HIGH = 30000  # 30m

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low

    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    def setDeltaRoi(self, delta):
        self.DELTA = delta

    def _check_input(
        self, roi, frame
    ):  # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!"
            )
        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA)

    # roi has to be list of ints
    def calc_spatials(self, depthData, roi, averaging_method=np.mean):
        depthFrame = depthData.getFrame()

        roi = self._check_input(
            roi, depthFrame
        )  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = np.array(  # Get centroid of the ROI
            [
                int((xmax + xmin) / 2),
                int((ymax + ymin) / 2),
            ]
        )

        K = self.calibData.getCameraIntrinsics(
            cameraId=self.depth_alignment_socket,
            resizeWidth=depthFrame.shape[1],
            resizeHeight=depthFrame.shape[0],
        )
        K = np.array(K)
        K_inv = np.linalg.inv(K)
        homogenous_coords = np.array([centroid[0], centroid[1], 1])
        spatial_coords = averageDepth * K_inv.dot(homogenous_coords)

        spatials = {
            "x": spatial_coords[0],
            "y": spatial_coords[1],
            "z": spatial_coords[2],
        }
        return spatials
