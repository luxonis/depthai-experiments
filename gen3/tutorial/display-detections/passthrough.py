import depthai as dai
import blobconverter
import cv2
import numpy as np

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class DisplayDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, rgb: dai.Node.Output, nn: dai.Node.Output) -> "DisplayDetections":
        self.link_args(rgb, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, rgb: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        color = (255, 0, 0)
        frame = rgb.getCvFrame()

        for detection in detections.detections:
            bbox = (np.array((detection.xmin, detection.ymin, detection.xmax, detection.ymax)) * frame.shape[0]).astype(int)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        cv2.imshow("Detections", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setInterleaved(False)
    cam.setIspScale(1, 3)  # 4k -> 720P
    # Crop video to match detection network
    cam.setPreviewSize(300, 300)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=5))
    cam.preview.link(nn.input)

    display = pipeline.create(DisplayDetections).build(
        rgb=cam.preview,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
