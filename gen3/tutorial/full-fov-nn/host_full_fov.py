import cv2
import depthai as dai
import numpy as np
from enum import Enum

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class MANIP_MODE(Enum):
    CROP, \
    LETTERBOX, \
    STRETCH = range(3)

class FullFOV(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.manip_mode = MANIP_MODE.CROP
        self.i = 0

    def build(self, isp: dai.Node.Output, crop_manip: dai.Node.Output, letterbox_manip: dai.Node.Output,
              stretch_manip: dai.Node.Output, nn_manip: dai.Node.Output, nn: dai.Node.Output
              , manip_config: dai.Node.Input) -> "FullFOV":
        self.link_args(isp, crop_manip, letterbox_manip, stretch_manip, nn_manip, nn)
        self.sendProcessingToPipeline(True)
        self._config = manip_config
        return self

    def process(self, isp: dai.ImgFrame, crop: dai.ImgFrame, letterbox: dai.ImgFrame, stretch: dai.ImgFrame,
                manip: dai.ImgFrame, detections: dai.ImgDetections) -> None:

        crop_frame = crop.getCvFrame()
        cv2.putText(crop_frame, "'a' to select crop", (20, 30), cv2.FONT_HERSHEY_SIMPLEX
                    , 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        letterbox_frame = letterbox.getCvFrame()
        cv2.putText(letterbox_frame, "'s' to select letterbox", (20, 30), cv2.FONT_HERSHEY_SIMPLEX
                    , 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        stretch_frame = stretch.getCvFrame()
        cv2.putText(stretch_frame, "'d' to select stretch", (20, 30), cv2.FONT_HERSHEY_SIMPLEX
                    , 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        detections_frame = self.displayDetections(manip.getCvFrame(), detections.detections, MANIP_MODE.STRETCH)

        key = cv2.waitKey(1)
        config = dai.ImageManipConfig()

        if self.manip_mode == MANIP_MODE.CROP:
            combined_frame = np.concatenate((detections_frame, letterbox_frame, stretch_frame), axis=1)
        elif self.manip_mode == MANIP_MODE.LETTERBOX:
            combined_frame = np.concatenate((crop_frame, detections_frame, stretch_frame), axis=1)
        else:
            combined_frame = np.concatenate((crop_frame, letterbox_frame, detections_frame), axis=1)

        cv2.imshow("Combined frame", combined_frame)
        cv2.imshow("ISP", self.displayDetections(isp.getCvFrame(), detections.detections, self.manip_mode))

        if key == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
        elif key == ord('a'):
            config.setKeepAspectRatio(True)
            config.setResize(300, 300)
            self._config.send(config)
            self.manip_mode = MANIP_MODE.CROP
            self.i = 5
            print("Switched to crop mode.")
        elif key == ord('s'):
            config.setKeepAspectRatio(True)
            config.setResizeThumbnail(300, 300)
            self._config.send(config)
            self.manip_mode = MANIP_MODE.LETTERBOX
            self.i = 5
            print("Switched to letterbox mode.")
        elif key == ord('d'):
            config.setKeepAspectRatio(False)
            config.setResize(300, 300)
            self._config.send(config)
            self.manip_mode = MANIP_MODE.STRETCH
            print("Switched to stretch mode.")

    def frameNorm(self, frame, bbox, manip_mode):
        # moves the bounding box to equalize the crop
        if manip_mode == MANIP_MODE.CROP:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1] - 204
            ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
            ret[::2] += 102
            return ret
        # stretches the bounding box to equalize the letterbox
        elif manip_mode == MANIP_MODE.LETTERBOX:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1]
            print(bbox)
            bbox = (bbox[0]
                    , 0.5 + (bbox[1]-0.5)*frame.shape[1]/frame.shape[0]
                    , bbox[2]
                    , 0.5 + (bbox[3]-0.5)*frame.shape[1]/frame.shape[0])
            print(bbox)
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        # resizes the bounding box based on the frame size
        else:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayDetections(self, frame, detections, manip_mode):
        color = (255, 0, 0)
        for detection in detections:
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax), manip_mode)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return frame