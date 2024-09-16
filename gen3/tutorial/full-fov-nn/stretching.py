import depthai as dai
import cv2
import numpy as np

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class StretchedFOV(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, isp: dai.Node.Output, stretch_manip: dai.Node.Output, nn: dai.Node.Output) -> "StretchedFOV":
        self.link_args(isp, stretch_manip, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, isp: dai.ImgFrame, stretch: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        cv2.imshow("Stretching", self.displayDetections(stretch.getCvFrame(), detections.detections))
        cv2.imshow("ISP", self.displayDetections(isp.getCvFrame(), detections.detections))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def frameNorm(self, frame, bbox):
        # resizes the bounding box based on the frame size
        normVals = np.full(4, frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

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

    stretch_manip = pipeline.create(dai.node.ImageManip)
    stretch_manip.setMaxOutputFrameSize(270000)
    stretch_manip.setKeepAspectRatio(False)
    stretch_manip.initialConfig.setResize(300, 300)
    cam.preview.link(stretch_manip.inputImage)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    stretch_manip.out.link(nn.input)

    cropped_fov = pipeline.create(StretchedFOV).build(
        isp=cam.isp,
        stretch_manip=stretch_manip.out,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
