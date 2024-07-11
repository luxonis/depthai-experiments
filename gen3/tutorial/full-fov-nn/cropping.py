import depthai as dai
import blobconverter
import cv2
import numpy as np

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class CroppedFOV(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, isp: dai.Node.Output, crop_manip: dai.Node.Output, nn: dai.Node.Output) -> "CroppedFOV":
        self.link_args(isp, crop_manip, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, isp: dai.ImgFrame, crop: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        cv2.imshow("Cropping", self.displayDetections(crop.getCvFrame(), detections.detections))
        cv2.imshow("ISP", self.displayDetections(isp.getCvFrame(), detections.detections))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def frameNorm(self, frame, bbox):
        # moves the bounding box to equalize the crop
        normVals = np.full(4, frame.shape[0])
        ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        ret[::2] += (frame.shape[1] - frame.shape[0])//2
        return ret

    def displayDetections(self, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return frame

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setInterleaved(False)
    cam.setIspScale(1, 5)  # 4056x3040 -> 812x608
    cam.setPreviewSize(812, 608)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    # Slightly lower FPS to avoid lag, as ISP takes more resources at 12MP
    cam.setFps(25)

    crop_manip = pipeline.create(dai.node.ImageManip)
    crop_manip.setMaxOutputFrameSize(270000)
    crop_manip.initialConfig.setResize(300, 300)
    cam.preview.link(crop_manip.inputImage)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=5))
    crop_manip.out.link(nn.input)

    cropped_fov = pipeline.create(CroppedFOV).build(
        isp=cam.isp,
        crop_manip=crop_manip.out,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
