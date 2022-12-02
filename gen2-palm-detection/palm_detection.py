import numpy as np

class PalmDetection:
    def decode(self, frame, nn_data):
        """
        Each palm detection is a tensor consisting of 19 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 7 key_points
            - confidence score
        :return:
        """
        if nn_data is None: return []
        shape = (128, 128)
        num_keypoints = 7
        min_score_thresh = 0.7
        anchors = np.load("anchors_palm.npy")

        # Run the neural network
        results = self.to_tensor_result(nn_data)

        raw_box_tensor = results.get("regressors").reshape(-1, 896, 18)  # regress
        raw_score_tensor = results.get("classificators").reshape(-1, 896, 1)  # classification

        detections = self.raw_to_detections(raw_box_tensor, raw_score_tensor, anchors, shape, num_keypoints)

        palm_coords = [
            self.frame_norm(frame, *obj[:4])
            for det in detections
            for obj in det
            if obj[-1] > min_score_thresh
        ]

        palm_confs = [
            obj[-1] for det in detections for obj in det if obj[-1] > min_score_thresh
        ]

        if len(palm_coords) == 0: return []

        nms = self.non_max_suppression(
            boxes=np.concatenate(palm_coords).reshape(-1, 4),
            probs=palm_confs,
            overlapThresh=0.1,
        )

        if nms is None: return []
        return nms

    def sigmoid(self, x):
        return (1.0 + np.tanh(0.5 * x)) * 0.5

    def decode_boxes(self, raw_boxes, anchors, shape, num_keypoints):
        """
        Converts the predictions into actual coordinates using the anchor boxes.
        Processes the entire batch at once.
        """
        boxes = np.zeros_like(raw_boxes)
        x_scale, y_scale = shape

        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / x_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / y_scale * anchors[:, 3]

        boxes[..., 1] = y_center - h / 2.0  # xmin
        boxes[..., 0] = x_center - w / 2.0  # ymin
        boxes[..., 3] = y_center + h / 2.0  # xmax
        boxes[..., 2] = x_center + w / 2.0  # ymax

        for k in range(num_keypoints):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = (
                raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
            )
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def raw_to_detections(self, raw_box_tensor, raw_score_tensor, anchors_, shape, num_keypoints):
        """

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        detection_boxes = self.decode_boxes(raw_box_tensor, anchors_, shape, num_keypoints)
        detection_scores = self.sigmoid(raw_score_tensor).squeeze(-1)
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i]
            scores = np.expand_dims(detection_scores[i], -1)
            output_detections.append(np.concatenate((boxes, scores), -1))
        return output_detections

    def non_max_suppression(self, boxes, probs=None, angles=None, overlapThresh=0.3):
        if len(boxes) == 0:
            return [], []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        if probs is not None:
            idxs = probs

        idxs = np.argsort(idxs)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
            )

        if angles is not None:
            return boxes[pick].astype("int"), angles[pick]
        return boxes[pick].astype("int")

    def to_tensor_result(self, packet):
        return {
            name: np.array(packet.getLayerFp16(name))
            for name in [tensor.name for tensor in packet.getRaw().tensors]
        }

    def frame_norm(self, frame, *xy_vals):
        """
        nn data, being the bounding box locations, are in <0..1> range -
        they need to be normalized with frame width/height

        :param frame:
        :param xy_vals: the bounding box locations
        :return:
        """
        return (
            np.clip(np.array(xy_vals), 0, 1)
            * np.array(frame.shape[:2] * (len(xy_vals) // 2))[::-1]
        ).astype(int)
