# Based on code from NanoSAM: https://github.com/NVIDIA-AI-IOT/nanosam

import numpy as np
import onnxruntime as ort
import cv2

class ONNXDecoder:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

    def _preprocess_points(self, points, image_size, size: int = 1024):
        scale = size / max(*image_size)
        points = points * scale
        return points

    def _run_mask_decoder(
        self, features, points=None, point_labels=None, mask_input=None
    ):
        if points is not None:
            assert point_labels is not None
            assert len(points) == len(point_labels)

        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256))
            has_mask_input = np.array([0])
        else:
            has_mask_input = np.array([1])

        result = self.session.run(
            None,
            {
                "image_embeddings": features.astype(np.float32),
                "point_coords": np.asarray([points], dtype=np.float32),
                "point_labels": np.asarray([point_labels], dtype=np.float32),
                "mask_input": mask_input.astype(np.float32),
                "has_mask_input": has_mask_input.astype(np.float32),
            },
        )
        iou_predictions, low_res_masks = result

        return iou_predictions, low_res_masks

    def _upscale_mask(self, mask, image_shape, size=256):
        # works for batch size 1 only
        mask = mask[0].transpose(1, 2, 0)

        mask = cv2.resize(mask, (image_shape))

        return mask

    def predict(
        self, features, points, point_labels, mask_input=None, image_size=(1024, 1024)
    ):
        points = self._preprocess_points(points, (image_size[1], image_size[0]))
        mask_iou, low_res_mask = self._run_mask_decoder(
            features, points, point_labels, mask_input
        )

        cv2.imshow("lowmask", (low_res_mask[0, 0] > 0).astype(np.uint8) * 255)

        hi_res_mask = self._upscale_mask(low_res_mask, (image_size[1], image_size[0]))

        return hi_res_mask, mask_iou, low_res_mask
