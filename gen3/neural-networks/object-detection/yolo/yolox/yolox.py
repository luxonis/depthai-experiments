import time
import cv2
import depthai as dai
import numpy as np
from fps_handler import FPSHandler


class YoloX(dai.node.HostNode):
    def __init__(self) -> None:
        self._fps = FPSHandler()
        super().__init__()


    def build(self, img_frames: dai.Node.Output, nn_data: dai.Node.Output, shape: int, label_map: list[str]) -> "YoloX":
        self._nn_data_q = nn_data.createOutputQueue(maxSize=4, blocking=True)
        self.sendProcessingToPipeline(True)
        self.link_args(img_frames)
        self._shape = shape
        self._label_map = label_map
        return self


    def process(self, img_frame: dai.ImgFrame) -> None:    
        frame = img_frame.getCvFrame()
        self._fps.next_iter()
        cv2.putText(frame, "Fps: {:.2f}".format(self._fps.fps()), (2, self._shape - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
        nn_data: dai.NNData = self._nn_data_q.tryGet()
        if nn_data:
            self._draw_nn_data(frame, nn_data)

        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _draw_nn_data(self, frame: np.ndarray, nn_data: dai.NNData) -> None:
        data = nn_data.getTensor('output').reshape(1, 3549, 85)
        predictions = self._demo_postprocess(data, (self._shape, self._shape), p6=False)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        dets = self._multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.3)

        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]

            for i in range(len(final_boxes)):
                bbox = final_boxes[i]
                score = final_scores[i]
                class_name = self._label_map[int(final_cls_inds[i])]

                if score >= 0.1:
                        # Limit the bounding box to 0..SHAPE
                    bbox[bbox > self._shape - 1] = self._shape - 1
                    bbox[bbox < 0] = 0
                    xy_min = (int(bbox[0]), int(bbox[1]))
                    xy_max = (int(bbox[2]), int(bbox[3]))
                        # Display detection's BB, label and confidence on the frame
                    cv2.rectangle(frame, xy_min , xy_max, (255, 0, 0), 2)
                    cv2.putText(frame, class_name, (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(score * 100)}%", (xy_min[0] + 10, xy_min[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)


    def _demo_postprocess(self, outputs: np.ndarray, img_size: tuple, p6=False) -> np.ndarray:
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

    
    def _multiclass_nms(self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float) -> np.ndarray:
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
                keep = self._nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)
    

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float):
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