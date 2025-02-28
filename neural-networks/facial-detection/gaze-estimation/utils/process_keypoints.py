import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended, ImgDetectionsExtended


class ProcessKeypointDetections(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        self.left_config_output = self.createOutput()
        self.right_config_output = self.createOutput()
        self.face_config_output = self.createOutput()

        self._w = 1
        self._h = 1

        self._target_w = 100
        self._target_h = 100

    def crop_eye(self, eye_center, face_w, face_h):
        eye_w = int(face_w * 0.2)
        eye_h = int(face_h * 0.2)
        eye_center = dai.Point2f(eye_center[0], eye_center[1])
        size = dai.Size2f(eye_w, eye_h)
        eye = dai.RotatedRect(eye_center, size, 0)
        eye = eye.denormalize(self.w, self.h)

        return eye

    def create_crop_cfg(
        self, rectangle: dai.RotatedRect, img_detections: ImgDetectionsExtended
    ):
        cfg = dai.ImageManipConfigV2()
        cfg.addCropRotatedRect(rectangle, normalizedCoords=False)
        cfg.setOutputSize(self.target_w, self.target_h)
        cfg.setReusePreviousImage(False)
        cfg.setTimestamp(img_detections.getTimestamp())
        cfg.setSequenceNum(img_detections.getSequenceNum())

        return cfg

    def run(self) -> None:
        while self.isRunning():
            img_detections = self.detections_input.get()
            detections = img_detections.detections

            left_configs_message = dai.MessageGroup()
            right_configs_message = dai.MessageGroup()
            face_configs_message = dai.MessageGroup()

            for i, detection in enumerate(detections):
                detection: ImgDetectionExtended = detection
                keypoints = detection.keypoints
                face_size = detection.rotated_rect.size
                face_w, face_h = face_size.width * self.w, face_size.height * self.h

                right_eye_center = keypoints[0]
                right_eye_center = [
                    int(right_eye_center.x * self.w),
                    int(right_eye_center.y * self.h),
                ]
                right_eye = self.crop_eye(right_eye_center, face_w, face_h)
                right_eye = right_eye.denormalize(self.w, self.h)
                right_cfg = self.create_crop_cfg(right_eye, img_detections)
                right_configs_message[str(i + 100)] = right_cfg

                left_eye_center = keypoints[1]
                left_eye_center = [
                    int(left_eye_center.x * self.w),
                    int(left_eye_center.y * self.h),
                ]
                left_eye = self.crop_eye(left_eye_center, face_w, face_h)
                left_eye = left_eye.denormalize(self.w, self.h)
                left_cfg = self.create_crop_cfg(left_eye, img_detections)
                left_configs_message[str(i + 100)] = left_cfg

                face_rect = detection.rotated_rect
                face_rect = face_rect.denormalize(self.w, self.h)
                face_crop = self.create_crop_cfg(face_rect, img_detections)
                face_configs_message[str(i + 100)] = face_crop

            left_configs_message.setSequenceNum(img_detections.getSequenceNum())
            left_configs_message.setTimestamp(img_detections.getTimestamp())

            right_configs_message.setSequenceNum(img_detections.getSequenceNum())
            right_configs_message.setTimestamp(img_detections.getTimestamp())

            face_configs_message.setSequenceNum(img_detections.getSequenceNum())
            face_configs_message.setTimestamp(img_detections.getTimestamp())

            self.face_config_output.send(face_configs_message)
            self.left_config_output.send(left_configs_message)
            self.right_config_output.send(right_configs_message)

    def set_target_size(self, w: int, h: int):
        """Set the target size for the output image."""
        self._target_w = w
        self._target_h = h

    def set_source_size(self, w: int, h: int):
        """Set the source size for the input image."""
        self._w = w
        self._h = h

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def target_w(self):
        return self._target_w

    @property
    def target_h(self):
        return self._target_h

    @w.setter
    def w(self, w):
        self._w = w

    @h.setter
    def h(self, h):
        self._h = h

    @target_w.setter
    def target_w(self, target_w):
        self._target_w = target_w

    @target_h.setter
    def target_h(self, target_h):
        self._target_h = target_h
