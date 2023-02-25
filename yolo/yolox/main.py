import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float16)
    return padded_img, r


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


syncNN = False

SHAPE = 416
labelMap = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

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


camera = p.create(dai.node.ColorCamera)
camera.setPreviewSize(SHAPE, SHAPE)
camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camera.setInterleaved(False)
camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath(str(Path("yolox_tiny.blob").resolve().absolute()))
nn.setNumInferenceThreads(2)
nn.input.setBlocking(True)

# Send camera frames to the host
camera_xout = p.create(dai.node.XLinkOut)
camera_xout.setStreamName("camera")
camera.preview.link(camera_xout.input)

# Send converted frames from the host to the NN
nn_xin = p.create(dai.node.XLinkIn)
nn_xin.setStreamName("nnInput")
nn_xin.out.link(nn.input)

# Send bounding boxes from the NN to the host via XLink
nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    qCamera = device.getOutputQueue(name="camera", maxSize=4, blocking=False)
    qNnInput = device.getInputQueue("nnInput", maxSize=4, blocking=False)
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=True)
    fps = FPSHandler()

    while True:
        inRgb = qCamera.get()
        frame = inRgb.getCvFrame()
        # Set these according to your dataset
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        image, ratio = preproc(frame, (SHAPE, SHAPE), mean, std)
        # NOTE: The model expects an FP16 input image, but ImgFrame accepts a list of ints only. I work around this by
        # spreading the FP16 across two ints
        image = list(image.tobytes())

        dai_frame = dai.ImgFrame()
        dai_frame.setHeight(SHAPE)
        dai_frame.setWidth(SHAPE)
        dai_frame.setData(image)
        qNnInput.send(dai_frame)

        if syncNN:
            in_nn = qNn.get()
        else:
            in_nn = qNn.tryGet()
        if in_nn is not None:
            fps.next_iter()
            cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, SHAPE - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        color=(255, 255, 255))

            data = np.array(in_nn.getLayerFp16('output')).reshape(1, 3549, 85)
            predictions = demo_postprocess(data, (SHAPE, SHAPE), p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4, None] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.3)

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]

                for i in range(len(final_boxes)):
                    bbox = final_boxes[i]
                    score = final_scores[i]
                    class_name = labelMap[int(final_cls_inds[i])]

                    if score >= 0.1:
                        # Limit the bounding box to 0..SHAPE
                        bbox[bbox > SHAPE - 1] = SHAPE - 1
                        bbox[bbox < 0] = 0
                        xy_min = (int(bbox[0]), int(bbox[1]))
                        xy_max = (int(bbox[2]), int(bbox[3]))
                        # Display detection's BB, label and confidence on the frame
                        cv2.rectangle(frame, xy_min, xy_max, (255, 0, 0), 2)
                        cv2.putText(frame, class_name, (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                                    255)
                        cv2.putText(frame, f"{int(score * 100)}%", (xy_min[0] + 10, xy_min[1] + 40),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            break
