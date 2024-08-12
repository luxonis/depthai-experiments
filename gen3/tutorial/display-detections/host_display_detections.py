import depthai as dai
import cv2
import numpy as np

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class DisplayDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, full: dai.Node.Output, square: dai.Node.Output
              , passthrough: dai.Node.Output, nn: dai.Node.Output) -> "DisplayDetections":
        self.link_args(full, square, passthrough, nn)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, full: dai.ImgFrame, square: dai.ImgFrame
                , passthrough: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        cv2.imshow("Passthrough", self.displayDetections(passthrough.getCvFrame(), detections.detections, False))
        cv2.imshow("Crop high res", self.displayDetections(square.getCvFrame(), detections.detections, False))
        cv2.imshow("Crop before inference", self.displayDetections(full.getCvFrame(), detections.detections, True))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()


    def frameNorm(self, frame, bbox, center):
        normVals = np.full(4, frame.shape[0])
        ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        # moves the bounding box to equalize the crop
        if center:
            ret[::2] += (frame.shape[1] - frame.shape[0])//2
        return ret


    def displayDetections(self, frame, detections, center):
        color = (255, 0, 0)
        for detection in detections:
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax), center)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return frame