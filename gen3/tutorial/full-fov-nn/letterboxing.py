import depthai as dai
import cv2
import numpy as np

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class LetterboxedFOV(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, isp: dai.Node.Output, letterbox_manip: dai.Node.Output, nn: dai.Node.Output) -> "LetterboxedFOV":
        self.link_args(isp, letterbox_manip, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, isp: dai.ImgFrame, letterbox: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        cv2.imshow("Letterboxing", self.displayDetections(letterbox.getCvFrame(), detections.detections))
        cv2.imshow("ISP", self.displayDetections(isp.getCvFrame(), detections.detections))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def frameNorm(self, frame, bbox):
        # stretches the bounding box to equalize the letterbox
        normVals = np.full(4, frame.shape[0])
        normVals[::2] = frame.shape[1]
        bbox = (bbox[0]
                , 0.5 + (bbox[1] - 0.5) * frame.shape[1] / frame.shape[0]
                , bbox[2]
                , 0.5 + (bbox[3] - 0.5) * frame.shape[1] / frame.shape[0])
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

    letterbox_manip = pipeline.create(dai.node.ImageManip)
    letterbox_manip.setMaxOutputFrameSize(270000)
    letterbox_manip.initialConfig.setResizeThumbnail(300, 300)
    cam.preview.link(letterbox_manip.inputImage)

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setNNArchive(nn_archive)
    letterbox_manip.out.link(nn.input)

    cropped_fov = pipeline.create(LetterboxedFOV).build(
        isp=cam.isp,
        letterbox_manip=letterbox_manip.out,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
