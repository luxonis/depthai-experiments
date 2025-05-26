import depthai as dai
from utils.img_annotation_helper import AnnotationHelper 
from typing import Any, Optional
import numpy as np
import time
from types import SimpleNamespace

config = SimpleNamespace(
    checkerboard_size=(10, 8),  # number of squares on the checkerboard
    square_size=0.02,           # size of a square in meters
    calibration_data_dir="calibration_data"  # directory relative to main.py
)


class CalibrationNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.calib = False
        self.checkerboard_size = config.checkerboard_size
        self.checkerboard_inner_size = (
            self.checkerboard_size[0] - 1,
            self.checkerboard_size[1] - 1,
        )
        self.square_size = config.square_size
        self.corners_world = np.zeros(
            (1, self.checkerboard_inner_size[0] * self.checkerboard_inner_size[1], 3),
            np.float32,
        )
        self.corners_world[0, :, :2] = np.mgrid[
            0:self.checkerboard_inner_size[0], 0:self.checkerboard_inner_size[1]
        ].T.reshape(-1, 2)
        self.corners_world *= self.square_size
        self.distortion_coef = np.zeros((1, 5))
        self.rot_vec = None
        self.trans_vec = None
        self.world_to_cam = None
        self.cam_to_world = None
        self.intrinsic_mat = None
        self.device = None
        self.mxid = None

        self.annotation_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.stream_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def _create_dai_color(
        self, r: float, g: float, b: float, a: float = 1.0
    ) -> dai.Color:
        color = dai.Color()
        color.r = r
        color.g = g
        color.b = b
        color.a = a
        return color

    def build(
        self,
        input: dai.Node.Output,
        intrinsic_mat,
        device: dai.Device,
    ) -> "CalibrationNode":
        self.intrinsic_mat = intrinsic_mat
        self.device = device
        self.mxid = device.getDeviceId()

        self.link_args(input)
        return self

    def _capture_still(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        if not self.calib:
            return None
        print(f"Capturing still for {self.mxid}")
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)

        start_time = time.time() * 1000
        in_still = None
        while in_still is None:
            in_still = self.still_out.tryGet()
            if time.time() * 1000 - start_time > timeout_ms:
                print("Did not receive still image - retrying")
                return self._capture_still(timeout_ms)
            time.sleep(0.1)

        still_rgb = cv2.imdecode(in_still.getFrame(), cv2.IMREAD_UNCHANGED)
        return still_rgb

    def process(
        self, frame_message: dai.ImgFrame
    ) -> None:
        annotations_builder = AnnotationHelper()

        annotations_builder.draw_text(
            text=f"This camera is chosen: {self.calib}",
            position=(0.02, 0.05),
            color=(0, 0, 0, 1),
            background_color=(1, 1, 1, 0.7),
            size=4,
        )
        annotations_builder.draw_text(
            text="Press 'c' to calibrate",
            position=(0.02, 0.1),
            color=(0, 0, 0, 1),
            background_color=(1, 1, 1, 0.7),
            size=4,
        )

        annotations = annotations_builder.build(
            frame_message.getTimestamp(), frame_message.getSequenceNum()
        )

        self.annotation_out.send(annotations)
