import math
import numpy as np
import depthai as dai
import cv2
from utility import TextHelper


class HostSpatialsCalc(dai.node.HostNode):
    # We need device object to get calibration data
    def __init__(self):
        # Values
        super().__init__()
        self.DELTA = 5
        self.THRESH_LOW = 200 # 20cm
        self.THRESH_HIGH = 30000 # 30m

        self.text = TextHelper()
        self.y = 200
        self.x = 300
        self.step = 3
        self.delta = 5
        self.setDeltaRoi(self.delta)

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])



    def build(self, disparity_frames: dai.Node.Output, depth_frames: dai.Node.Output, calibData : dai.CalibrationHandler) -> "HostSpatialsCalc":
        self.calibData = calibData

        self.link_args(disparity_frames, depth_frames)
        self.sendProcessingToPipeline(True)
        print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")
        return self


    def process(self, disparity : dai.ImgFrame, depth : dai.ImgFrame) -> None:
        # Calculate spatial coordiantes from depth frame
        #TODO: make spatials calculation separate node
        spatials, centroid = self.calc_spatials(depth, (self.x, self.y)) # centroid == x/y in our case

        disp = disparity.getCvFrame()
        self.text.rectangle(disp, (self.x-self.delta, self.y-self.delta), (self.x+self.delta, self.y+self.delta))
        self.text.putText(disp, "X: " + ("{:.1f}m".format(spatials['x']/1000) if not math.isnan(spatials['x']) else "--"), (self.x + 10, self.y + 20))
        self.text.putText(disp, "Y: " + ("{:.1f}m".format(spatials['y']/1000) if not math.isnan(spatials['y']) else "--"), (self.x + 10, self.y + 35))
        self.text.putText(disp, "Z: " + ("{:.1f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (self.x + 10, self.y + 50))

        disparity.setCvFrame(disp, dai.ImgFrame.Type.BGR888i)
        self.output.send(disparity)

        #TODO: Add keyboard reader node
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.stopPipeline()
        elif key == ord('w'):
            self.y -= self.step
        elif key == ord('a'):
            self.x -= self.step
        elif key == ord('s'):
            self.y += self.step
        elif key == ord('d'):
            self.x += self.step
        elif key == ord('r'): # Increase Delta
            if self.delta < 50:
                self.delta += 1
                self.setDeltaRoi(self.delta)
        elif key == ord('f'): # Decrease Delta
            if 3 < self.delta:
                self.delta -= 1
                self.setDeltaRoi(self.delta)


    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low


    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low
    
    
    def setDeltaRoi(self, delta):
        self.DELTA = delta
    
        
    def _check_input(self, roi, frame): # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5 # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x-self.DELTA,y-self.DELTA,x+self.DELTA,y+self.DELTA)

    
    def _calc_angle(self, frame, offset, HFOV):
        return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    
    # roi has to be list of ints
    def calc_spatials(self, depthData, roi, averaging_method=np.mean):

        depthFrame = depthData.getFrame()

        roi = self._check_input(roi, depthFrame) # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        # Required information for calculating spatial coordinates on the host
        HFOV = np.deg2rad(self.calibData.getFov(dai.CameraBoardSocket(depthData.getInstanceNum()), useSpec=False))

        if inRange.any():
            averageDepth = averaging_method(depthROI[inRange])
        else:
            averageDepth = np.nan
        
        centroid = { # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2) # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2) # middle of the depth img height
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos, HFOV)
        angle_y = self._calc_angle(depthFrame, bb_y_pos, HFOV)

        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid
