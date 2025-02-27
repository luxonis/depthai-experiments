import numpy as np
import depthai as dai
from numpy_buffer import NumpyBuffer

from utils import frame_norm, to_planar, copy_timestamps


class FaceProcessing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output_landmarks = dai.Node.Output(self)
        self.output_pose = dai.Node.Output(self)
        self.output_face = dai.Node.Output(self)

    def build(
        self, img_frames: dai.Node.Output, bboxes: dai.Node.Output
    ) -> "FaceProcessing":
        self.link_args(img_frames, bboxes)
        return self

    def process(self, img_frame: dai.ImgFrame, bboxes: dai.Buffer) -> None:
        assert isinstance(bboxes, NumpyBuffer)
        frame = img_frame.getCvFrame()
        bboxes_data = bboxes.getData()

        land_data = dai.NNData()
        copy_timestamps(img_frame, land_data)

        pose_data = dai.NNData()
        copy_timestamps(img_frame, pose_data)

        face_buffer = NumpyBuffer(np.zeros(0), img_frame)

        if bboxes_data.size == 0:
            self._send_outputs(land_data, pose_data, face_buffer)
            return

        land_results: list[list] = []
        pose_results: list[list] = []
        face_results: list[list] = []
        for raw_bbox in bboxes_data:
            bbox = frame_norm(frame, raw_bbox)
            det_frame = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            land_results.append(to_planar(det_frame, (48, 48)))
            pose_results.append(to_planar(det_frame, (60, 60)))
            face_results.append(bbox)

        land_data.addTensor("0", np.array(land_results))
        pose_data.addTensor("data", np.array(pose_results))
        face_buffer.setData(np.array(face_results), img_frame)

        self._send_outputs(land_data, pose_data, face_buffer)

    def _send_outputs(
        self, land_data: dai.NNData, pose_data: dai.NNData, face_buffer: NumpyBuffer
    ) -> None:
        self.output_landmarks.send(land_data)
        self.output_pose.send(pose_data)
        self.output_face.send(face_buffer)
