import cv2
import depthai as dai


class StereoConfigHandler:

    class Trackbar:
        def __init__(self, trackbarName, windowName, minValue, maxValue, defaultValue, handler):
            self.min = minValue
            self.max = maxValue
            self.windowName = windowName
            self.trackbarName = trackbarName
            cv2.createTrackbar(trackbarName, windowName, minValue, maxValue, handler)
            cv2.setTrackbarPos(trackbarName, windowName, defaultValue)

        def set(self, value):
            if value < self.min:
                value = self.min
                print(f'{self.trackbarName} min value is {self.min}')
            if value > self.max:
                value = self.max
                print(f'{self.trackbarName} max value is {self.max}')
            cv2.setTrackbarPos(self.trackbarName, self.windowName, value)

    newConfig = False
    config = None
    trConfidence = list()
    trLrCheck = list()
    trCostAggregationP1 = list()
    trCostAggregationP2 = list()
    trOutlierRemovalThreshold = list()
    trOutlierCensusThreshold = list()
    trOutlierDiffThreshold = list()

    def trackbarConfidence(value):
        StereoConfigHandler.config.costMatching.confidenceThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trConfidence:
            tr.set(value)

    def trackbarLrCheckThreshold(value):
        StereoConfigHandler.config.algorithmControl.leftRightCheckThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trLrCheck:
            tr.set(value)

    def trackbarCostAggregationP1(value):
        StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP1 = value
        StereoConfigHandler.config.costAggregation.verticalPenaltyCostP1 = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trCostAggregationP1:
            tr.set(value)

    def trackbarCostAggregationP2(value):
        StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP2 = value
        StereoConfigHandler.config.costAggregation.verticalPenaltyCostP2 = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trCostAggregationP2:
            tr.set(value)

    def trackbarOutlierRemovalThreshold(value):
        StereoConfigHandler.config.algorithmControl.outlierRemoveThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trOutlierRemovalThreshold:
            tr.set(value)

    def trackbarOutlierCensusThreshold(value):
        StereoConfigHandler.config.algorithmControl.outlierCensusThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trOutlierCensusThreshold:
            tr.set(value)

    def trackbarOutlierDiffThreshold(value):
        StereoConfigHandler.config.algorithmControl.outlierDiffThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trOutlierDiffThreshold:
            tr.set(value)


    def handleKeypress(key, stereoDepthConfigInQueue):
        if key == ord('l'):
            StereoConfigHandler.newConfig = True
            currentCfg = StereoConfigHandler.config.costAggregation.localAggregationMode
            nextCfg = cfgSettings[(cfgSettings.index(currentCfg)+1) % len(cfgSettings)]
            print(f"Changing local aggregation to {nextCfg.name} from {currentCfg.name}")
            StereoConfigHandler.config.costAggregation.localAggregationMode = nextCfg
        elif key == ord('r'):
            StereoConfigHandler.newConfig = True
            currentCfg = StereoConfigHandler.config.algorithmControl.replaceInvalidDisparity
            nextCfg = not currentCfg
            print(f"Changing replace invalid disparity to {nextCfg} from {currentCfg}")
            StereoConfigHandler.config.algorithmControl.replaceInvalidDisparity = nextCfg

        StereoConfigHandler.sendConfig(stereoDepthConfigInQueue)

    def sendConfig(stereoDepthConfigInQueue):
        if StereoConfigHandler.newConfig:
            StereoConfigHandler.newConfig = False
            configMessage = dai.StereoDepthConfig()
            configMessage.set(StereoConfigHandler.config)
            configMessage.setData([1]) # NOTE workaround
            stereoDepthConfigInQueue.send(configMessage)

    def updateDefaultConfig(config):
        StereoConfigHandler.config = config

    def registerWindow(stream):
        cv2.namedWindow(stream, cv2.WINDOW_NORMAL)

        StereoConfigHandler.trConfidence.append(StereoConfigHandler.Trackbar('Disparity confidence', stream, 0, 255, StereoConfigHandler.config.costMatching.confidenceThreshold, StereoConfigHandler.trackbarConfidence))
        StereoConfigHandler.trCostAggregationP1.append(StereoConfigHandler.Trackbar('Cost aggregation P1', stream, 0, 500, StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP1, StereoConfigHandler.trackbarCostAggregationP1))
        StereoConfigHandler.trCostAggregationP2.append(StereoConfigHandler.Trackbar('Cost aggregation P2', stream, 0, 500, StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP2, StereoConfigHandler.trackbarCostAggregationP2))
        StereoConfigHandler.trLrCheck.append(StereoConfigHandler.Trackbar('Left-right check threshold', stream, 0, 255, StereoConfigHandler.config.algorithmControl.leftRightCheckThreshold, StereoConfigHandler.trackbarLrCheckThreshold))
        StereoConfigHandler.trOutlierRemovalThreshold.append(StereoConfigHandler.Trackbar('Outlier removal threshold', stream, 0, 49, StereoConfigHandler.config.algorithmControl.outlierRemoveThreshold, StereoConfigHandler.trackbarOutlierRemovalThreshold))
        StereoConfigHandler.trOutlierCensusThreshold.append(StereoConfigHandler.Trackbar('Outlier census threshold', stream, 32, 255, StereoConfigHandler.config.algorithmControl.outlierCensusThreshold, StereoConfigHandler.trackbarOutlierCensusThreshold))
        StereoConfigHandler.trOutlierDiffThreshold.append(StereoConfigHandler.Trackbar('Outlier difference threshold', stream, 1, 96, StereoConfigHandler.config.algorithmControl.outlierDiffThreshold, StereoConfigHandler.trackbarOutlierDiffThreshold))

    def __init__(self, config):

        StereoConfigHandler.config = config