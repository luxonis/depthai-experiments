import cv2
import depthai as dai
import numpy as np

from depthai_sdk import OakCamera

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


def callback(packet):
    frame = packet.frame
    nn_data = packet.img_detections

    bb = np.array(nn_data.getLayerFp16('Identity')).reshape(25, 4)
    label = nn_data.getLayerInt32('Identity_1')
    conf = nn_data.getLayerFp16('Identity_2')

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
            cv2.putText(frame, coco_90[label[i]], (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(conf[i] * 100)}%", (xy_min[0] + 10, xy_min[1] + 40), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, 255)

    cv2.imshow('EfficientDet', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.node.setFp16(True)

    nn = oak.create_nn('models/efficientdet_lite0_2021.3_6shaves.blob', color)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
