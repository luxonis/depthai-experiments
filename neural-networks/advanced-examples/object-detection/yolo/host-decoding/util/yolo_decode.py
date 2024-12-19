from typing import List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def nms(dets: np.ndarray, nms_thresh: float = 0.5) -> List[int]:
    """Non-maximum suppression."""
    thresh = nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def xywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding box coordinates from (x_center, y_center, width, height) to
    (x_min, y_min, x_max, y_max)."""
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("Bounding boxes must be a numpy array.")
    if len(bboxes.shape) != 2:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )
    if bboxes.shape[1] != 4:
        raise ValueError(
            "Bounding boxes must be of shape (N, 4). Got shape {bboxes.shape}."
        )

    xyxy_bboxes = np.zeros_like(bboxes)
    xyxy_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_min = x - w/2
    xyxy_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_min = y - h/2
    xyxy_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x_max = x + w/2
    xyxy_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y_max = y + h/2

    return xyxy_bboxes


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    classes: List = None,
    num_classes: int = 1,
    agnostic: bool = False,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> List[np.ndarray]:
    """Performs Non-Maximum Suppression (NMS) on inference results."""

    # Detection: 4 (bbox) + 1 (objectness) = 5
    num_classes_check = prediction.shape[2] - 5

    nm = prediction.shape[2] - num_classes - 5
    pred_candidates = prediction[..., 4] > conf_thres  # candidates

    # Check the parameters.
    assert (
        num_classes == num_classes_check
    ), f"Number of classes {num_classes} does not match the model {num_classes_check}"
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    output = [np.zeros((0, 6 + nm))] * prediction.shape[0]

    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh_to_xyxy(x[:, :4])
        cls = x[:, 5 : 5 + num_classes]
        other = x[:, 5 + num_classes :]  # Either kpts or pos

        class_idx = np.expand_dims(cls.argmax(1), 1)
        conf = cls.max(1, keepdims=True)
        x = np.concatenate((box, conf, class_idx, other), 1)[
            conf.flatten() > conf_thres
        ]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort()[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = (
            x[:, :4] + class_offset,
            x[:, 4][..., np.newaxis],
        )  # boxes (offset by class), scores
        keep_box_idx = np.array(
            nms(np.hstack((boxes, scores)).astype(np.float32, copy=False), iou_thres)
        )

        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]

    return output


def make_grid_numpy(ny: int, nx: int, na: int) -> np.ndarray:
    """Create a grid of shape (1, na, ny, nx, 2)"""
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.stack((xv, yv), 2).reshape(1, na, ny, nx, 2)


def parse_yolo_output(
    out: np.ndarray,
    stride: int,
    num_outputs: int,
) -> np.ndarray:
    """Parse a single channel output of an YOLO model."""
    bs, _, ny, nx = out.shape  # bs - batch size, ny|nx - y and x of grid cells

    grid = make_grid_numpy(ny, nx, 1)

    out = out.reshape(bs, 1, num_outputs, ny, nx).transpose((0, 1, 3, 4, 2))

    x1y1 = grid - out[..., 0:2] + 0.5
    x2y2 = grid + out[..., 2:4] + 0.5
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    out[..., 0:2] = c_xy * stride
    out[..., 2:4] = wh * stride

    out = out.reshape(bs, -1, num_outputs)

    return out


def parse_yolo_outputs(
    outputs: List[np.ndarray], strides: List[int], num_outputs: int
) -> np.ndarray:
    """Parse all outputs of an YOLO model (all channels)."""
    output = None

    for out_head, stride in zip(outputs, strides):
        out = parse_yolo_output(
            out_head,
            stride,
            num_outputs,
        )
        output = out if output is None else np.concatenate((output, out), axis=1)

    return output


def decode_yolo_output(
    yolo_outputs: List[np.ndarray],
    strides: List[int],
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    num_classes: int = 1,
) -> np.ndarray:
    """Decode the output of an YOLO instance segmentation or pose estimation model."""
    num_outputs = num_classes + 5
    output = parse_yolo_outputs(yolo_outputs, strides, num_outputs)
    output_nms = non_max_suppression(
        output,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        num_classes=num_classes,
    )[0]

    return output_nms
