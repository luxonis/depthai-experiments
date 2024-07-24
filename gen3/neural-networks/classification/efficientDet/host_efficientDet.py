import time
import depthai as dai
import cv2
import numpy as np

COCO_90 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "12", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "26", "backpack", "umbrella", "29", "30", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "45", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "66", "dining table", "68", "69", "toilet", "71", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "83", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
CONF_THRESHOLD = 0.4
SHAPE = (3, 320, 320)

class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


class EfficientDet(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.fps = FPSHandler()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "EfficientDet":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        # Model needs FP16 so we have to convert color frame back to U8 on the host
        frame = np.array(preview.getData()).view(np.float16).reshape(SHAPE).transpose(1, 2, 0).astype(np.uint8).copy()

        self.fps.next_iter()
        cv2.putText(frame, "Fps: {:.2f}".format(self.fps.fps())
                    , (2, 316), cv2.FONT_HERSHEY_TRIPLEX, 0.4,color=(255, 255, 255))

        bb = nn.getTensor("Identity").flatten().reshape(25, 4)
        label = nn.getTensor("Identity_1").flatten().astype(np.int32)
        conf = nn.getTensor("Identity_2").flatten()

        for i in range(len(conf)):
            if CONF_THRESHOLD < conf[i]:
                bb_det = bb[i]
                # Limit the bounding box to 0..1
                bb_det[bb_det > 1] = 1
                bb_det[bb_det < 0] = 0
                xy_min = (int(bb_det[1]*320), int(bb_det[0]*320))
                xy_max = (int(bb_det[3]*320), int(bb_det[2]*320))

                cv2.rectangle(frame, xy_min , xy_max, (255, 0, 0), 2)
                cv2.putText(frame, COCO_90[label[i]], (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf[i] * 100)}%", (xy_min[0] + 10, xy_min[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
