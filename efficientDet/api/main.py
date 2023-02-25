import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

CONF_THRESHOLD = 0.4
SHAPE = 320
coco_90 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "12", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "26", "backpack", "umbrella", "29", "30", "handbag", "tie",
           "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
           "skateboard", "surfboard", "tennis racket", "bottle", "45", "wine glass", "cup", "fork", "knife", "spoon",
           "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "couch", "potted plant", "bed", "66", "dining table", "68", "69", "toilet", "71", "tv", "laptop",
           "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "83",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_3)


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


camRgb = p.create(dai.node.ColorCamera)
camRgb.setPreviewSize(SHAPE, SHAPE)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFp16(True)  # Model requires FP16 input

# NN that detects faces in the image
nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath(str(Path("models/efficientdet_lite0_2021.3_6shaves.blob").resolve().absolute()))
nn.setNumInferenceThreads(2)
camRgb.preview.link(nn.input)

# Send bouding box from the NN to the host via XLink
nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)
# Send rgb frames to the host
rgb_xout = p.create(dai.node.XLinkOut)
rgb_xout.setStreamName("rgb")
nn.passthrough.link(rgb_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    fps = FPSHandler()
    shape = (3, SHAPE, SHAPE)

    while True:
        inRgb = qRgb.get()
        # Model needs FP16 so we have to convert color frame back to U8 on the host
        frame = np.array(inRgb.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8).copy()

        in_nn = qNn.tryGet()
        if in_nn is not None:
            fps.next_iter()
            cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, SHAPE - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        color=(255, 255, 255))

            # You can use the line below to print all the (output) layers and their info
            # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in in_nn.getAllLayers()]

            bb = np.array(in_nn.getLayerFp16('Identity')).reshape(25, 4)
            label = in_nn.getLayerInt32('Identity_1')
            conf = in_nn.getLayerFp16('Identity_2')

            for i in range(len(conf)):
                if CONF_THRESHOLD < conf[i]:
                    bb_det = bb[i]
                    # Limit the bounding box to 0..1
                    bb_det[bb_det > 1] = 1
                    bb_det[bb_det < 0] = 0
                    xy_min = (int(bb_det[1] * SHAPE), int(bb_det[0] * SHAPE))
                    xy_max = (int(bb_det[3] * SHAPE), int(bb_det[2] * SHAPE))
                    # Display detection's BB, label and confidence on the frame
                    cv2.rectangle(frame, xy_min, xy_max, (255, 0, 0), 2)
                    cv2.putText(frame, coco_90[label[i]], (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX,
                                0.5, 255)
                    cv2.putText(frame, f"{int(conf[i] * 100)}%", (xy_min[0] + 10, xy_min[1] + 40),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            break
