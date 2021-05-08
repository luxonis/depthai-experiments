from pathlib import Path
import numpy as np
import cv2
import depthai as dai
import time

CONF_THRESHOLD = 0.4
SHAPE = 320
coco_90 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "12", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "26", "backpack", "umbrella", "29", "30", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "45", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "66", "dining table", "68", "69", "toilet", "71", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "83", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

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

camRgb = p.createColorCamera()
camRgb.setPreviewSize(SHAPE, SHAPE)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFp16(True)

# Send bouding box from the NN to host via XLink
rgb_xout = p.createXLinkOut()
rgb_xout.setStreamName("rgb")
camRgb.preview.link(rgb_xout.input)

# NN that detects faces in the image
nn = p.createNeuralNetwork()
nn.setBlobPath(str(Path("models/efficientdet_lite0.blob").resolve().absolute()))
nn.setNumInferenceThreads(2)
camRgb.preview.link(nn.input)

# Send bouding box from the NN to host via XLink
nn_xout = p.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    device.startPipeline()
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    fps = FPSHandler()

    while True:
        inRgb = qRgb.get()
        shape = (3, SHAPE, SHAPE)
        data = inRgb.getData()
        # TODO: FIx this mess
        frame = np.array(data).astype(np.uint8).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8).copy()

        data = qNn.tryGet()

        if data is not None:
            # for layer in data.getAllLayers(): print(f"Layer name: {layer.name}, Type: {layer.dataType}, Dimensions: {layer.dims}, Offset: {layer.offset}")
            fps.next_iter()
            cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, SHAPE - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            bb = np.array(data.getLayerFp16('Identity')).reshape(25, 4)
            label = data.getLayerInt32('Identity_1')
            conf = data.getLayerFp16('Identity_2')

            for i in range(len(conf)):
                if CONF_THRESHOLD < conf[i]:
                    bb_det = bb[i]
                    # print("original BB output:", bb_det)
                    bb_det[bb_det > 1] = 1
                    bb_det[bb_det < 0] = 0
                    # print("min/max limited BB:", bb_det)
                    xy_min = (int(bb_det[1]*SHAPE), int(bb_det[0]*SHAPE))
                    xy_max = (int(bb_det[3]*SHAPE), int(bb_det[2]*SHAPE))
                    cv2.rectangle(frame, xy_min , xy_max, (255, 0, 0), 2)
                    cv2.putText(frame, coco_90[label[i]], (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(conf[i] * 100)}%", (xy_min[0] + 10, xy_min[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            break
