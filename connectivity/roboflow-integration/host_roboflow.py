import depthai as dai
import numpy as np
import cv2
import time

from concurrent.futures import ThreadPoolExecutor
from uploader import RoboflowUploader

LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

class Roboflow(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        # Executor to handle uploads asynchronously
        # For real-time uploads at ~10Hz we spawn 40 threads
        self.executor = ThreadPoolExecutor(max_workers=40)
        self.last_upload_time = time.monotonic()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, uploader: RoboflowUploader
              , target_resolution: tuple[int, int], auto_interval: float, auto_threshold: float) -> "Roboflow":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        self.uploader = uploader
        self.target_res = target_resolution
        self.autoupload = (auto_interval != 0 and auto_threshold != 0)
        self.auto_interval = auto_interval
        self.auto_threshold = auto_threshold
        return self

    def process(self, preview: dai.ImgFrame, dets: dai.ImgDetections) -> None:
        frame = preview.getCvFrame()
        shape = (frame.shape[1], frame.shape[0])

        frame_with_boxes = overlay_boxes(frame, dets.detections)
        cv2.imshow("Roboflow Demo", frame_with_boxes)

        dt = time.monotonic() - self.last_upload_time

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

        elif key == 13:
            # If enter is pressed, upload all detections without thresholding
            labels, bboxes = parse_dets(dets.detections, shape, confidence_thr=0.0)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            print("Enter pressed. Uploading grabbed frame!")
            self.executor.submit(self.uploader.upload, frame_rgb, labels, bboxes)

        elif self.autoupload and dt > self.auto_interval:
            # Auto-upload annotations with confidence above self.auto_threshold every self.auto_interval seconds
            labels, bboxes = parse_dets(dets.detections, shape, confidence_thr=self.auto_threshold)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if len(bboxes) > 0:
                print(f"Auto-uploading grabbed frame with {len(bboxes)} annotations!")
                self.executor.submit(self.uploader.upload, frame_rgb, labels, bboxes)
            else:
                print(f"No detections with confidence above {self.auto_threshold}. Not uploading!")

            self.last_upload_time = time.monotonic()


def parse_dets(detections, image_shape, confidence_thr):
    width, height = image_shape
    labels = [LABELS[d.label] for d in detections if d.confidence > confidence_thr]

    bboxes = [[width * d.xmin, height * d.ymin, width * d.xmax, height * d.ymax]
              for d in detections if d.confidence > confidence_thr]

    return labels, bboxes

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def overlay_boxes(frame, detections):
    # Overlay on a copy of image to keep the original
    frame = frame.copy()
    BLUE = (255, 0, 0)

    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, BLUE)
        cv2.putText(frame,f"{int(detection.confidence * 100)}%",(bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, BLUE)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BLUE, 2)

    return frame
