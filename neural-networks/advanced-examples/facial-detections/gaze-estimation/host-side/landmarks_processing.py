import numpy as np
import depthai as dai

from utils import copy_timestamps, frame_norm, to_planar
from numpy_buffer import NumpyBuffer


class LandmarksProcessing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output_gaze = dai.Node.Output(self, possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.NNData, True)])
        self.output_pose = dai.Node.Output(self)
        self.output_left_bbox = dai.Node.Output(self)
        self.output_right_bbox = dai.Node.Output(self)
        self.output_nose = dai.Node.Output(self)

    
    def build(self, 
              img_frames: dai.Node.Output, 
              landmarks_nn: dai.Node.Output, 
              face_bboxes: dai.Node.Output, 
              poses_nn: dai.Node.Output
        ) -> "LandmarksProcessing":
        self.link_args(img_frames, landmarks_nn, face_bboxes, poses_nn)
        return self
    

    def process(self, 
                img_frame: dai.ImgFrame, 
                landmark_nn: dai.NNData,
                face_bbox_buffer: dai.Buffer,
                pose_nn: dai.NNData
            ) -> None:
        assert(isinstance(face_bbox_buffer, NumpyBuffer))
        frame: np.ndarray = img_frame.getCvFrame()
        left_bbox_buffer = NumpyBuffer(np.empty(0), img_frame)
        right_bbox_buffer = NumpyBuffer(np.empty(0), img_frame)
        nose_buffer = NumpyBuffer(np.empty(0), img_frame)
        pose_buffer = NumpyBuffer(np.empty(0), img_frame)
        gaze_data = dai.NNData()
        copy_timestamps(img_frame, gaze_data)

        if not landmark_nn.getAllLayerNames():
            self._send_outputs(gaze_data, left_bbox_buffer, right_bbox_buffer, nose_buffer, pose_buffer)
            return

        land_nn_output: np.ndarray = landmark_nn.getFirstTensor()
        batch_size = land_nn_output.shape[0]
        left_bboxes = []
        right_bboxes = []
        noses = []
        poses = []
        left_eye_imgs = []
        right_eye_imgs = []
        head_poses = []
        for batch_index in range(batch_size):
            land_in = land_nn_output[batch_index].flatten()
            face_bbox = face_bbox_buffer.getData()[batch_index]
            left = face_bbox[0]
            top = face_bbox[1]
            face_frame = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            land_data = frame_norm(face_frame, land_in)
            land_data[::2] += left
            land_data[1::2] += top

            left_bbox = self._padded_point(land_data[:2], padding=30, frame_shape=frame.shape)
            if left_bbox is None:
                print("Point for left eye is corrupted, skipping nn result...")
                return
            left_bboxes.append(left_bbox)

            right_bbox = self._padded_point(land_data[2:4], padding=30, frame_shape=frame.shape)
            if right_bbox is None:
                print("Point for right eye is corrupted, skipping nn result...")
                return
            right_bboxes.append(right_bbox)

            nose = land_data[4:6]
            noses.append(nose)

            left_img = frame[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
            right_img = frame[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
            

            # The output of  pose_nn is in YPR  format, which is the required sequence input for pose in  gaze
            # https://docs.openvinotoolkit.org/2020.1/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
            # https://docs.openvinotoolkit.org/latest/omz_models_model_gaze_estimation_adas_0002.html
            # ... three head pose angles – (yaw, pitch, and roll) ...
            pose = np.array([
                    pose_nn.getTensor('angle_y_fc')[batch_index][0],
                    pose_nn.getTensor('angle_p_fc')[batch_index][0],
                    pose_nn.getTensor('angle_r_fc')[batch_index][0]
                    ])
            poses.append(pose)

            left_eye_imgs.append(to_planar(left_img, (60, 60)))
            right_eye_imgs.append(to_planar(right_img, (60, 60)))
            head_poses.append(pose)

        left_bbox_buffer.setData(np.array(left_bboxes), img_frame)
        right_bbox_buffer.setData(np.array(right_bboxes), img_frame)
        nose_buffer.setData(np.array(noses), img_frame)
        pose_buffer.setData(np.array(poses), img_frame)

        gaze_data.addTensor("left_eye_image", np.array(left_eye_imgs))
        gaze_data.addTensor("right_eye_image", np.array(right_eye_imgs))
        gaze_data.addTensor("head_pose_angles", np.array(head_poses))
        
        self._send_outputs(gaze_data, left_bbox_buffer, right_bbox_buffer, nose_buffer, pose_buffer)

    
    def _send_outputs(self, gaze_data, left_bbox_buffer, right_bbox_buffer, nose_buffer, pose_buffer):
        self.output_gaze.send(gaze_data)
        self.output_left_bbox.send(left_bbox_buffer)
        self.output_right_bbox.send(right_bbox_buffer)
        self.output_nose.send(nose_buffer)
        self.output_pose.send(pose_buffer)


    def _padded_point(self, point, padding, frame_shape=None):
        if frame_shape is None:
            return [
                point[0] - padding,
                point[1] - padding,
                point[0] + padding,
                point[1] + padding
            ]
        else:
            def norm(val, dim):
                return max(0, min(val, dim))
            if np.any(point - padding > frame_shape[:2]) or np.any(point + padding < 0):
                print(f"Unable to create padded box for point {point} with padding {padding} and frame shape {frame_shape[:2]}")
                return None

            return [
                norm(point[0] - padding, frame_shape[0]),
                norm(point[1] - padding, frame_shape[1]),
                norm(point[0] + padding, frame_shape[0]),
                norm(point[1] + padding, frame_shape[1])
            ]