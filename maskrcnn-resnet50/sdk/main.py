import blobconverter
import cv2
import depthai as dai
import numpy as np

from depthai_sdk import OakCamera, ResizeMode
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox

NN_SHAPE = 300, 300
COLORS = np.random.random(size=(256, 3)) * 256
THRESHOLD = 0.5
REGION_THRESHOLD = 0.5
LABEL_MAP = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
    "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "mirror", "diningtable", "window", "desk", "toilet",
    "door", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def to_tensor(packet):
    """
    Converts NN packet to dict, with each key being output tensor name and each value being correctly reshaped and converted results array

    Useful as a first step of processing NN results for custom neural networks

    Args:
        packet (depthai.NNData): Packet returned from NN node

    Returns:
        dict: Dict containing prepared output tensors
    """
    data = {}
    for tensor in packet.getRaw().tensors:
        if tensor.dataType == dai.TensorInfo.DataType.INT:
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(tensor.dims)
        else:
            print("Unsupported tensor layer type: {}".format(tensor.dataType))
    return data


def callback(packet):
    frame = packet.frame
    nn_data = packet.img_detections

    tensors = to_tensor(nn_data)

    boxes = np.squeeze(tensors["DetectionOutput_647"])
    masks = tensors["Sigmoid_733"]
    for i, box in enumerate(boxes):
        if box[0] == -1:
            break

        cls = int(box[1])
        prob = box[2]

        if prob < THRESHOLD:
            continue

        bbox = NormalizeBoundingBox(NN_SHAPE, resize_mode=ResizeMode.LETTERBOX).normalize(frame, box[-4:])
        cv2.rectangle(frame, (bbox[0], bbox[1] - 15), (bbox[2], bbox[1]), COLORS[cls], -1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[cls], 1)
        cv2.putText(frame, f"{LABEL_MAP[cls - 1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 2)
        cv2.putText(frame, f"{LABEL_MAP[cls - 1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)

        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        mask = cv2.resize(masks[i, cls], (bbox_w, bbox_h))
        mask = mask > REGION_THRESHOLD

        roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        roi[mask] = roi[mask] * 0.6 + COLORS[cls] * 0.4
        frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi

    cv2.imshow('Mask-RCNN ResNet-50', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.config_color_camera(color_order=dai.ColorCameraProperties.ColorOrder.RGB)

    nn_path = blobconverter.from_zoo("mask_rcnn_resnet50_coco_300x300", shaves=6, zoo_type="depthai")
    nn = oak.create_nn(nn_path, color)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
