import cv2
import depthai as dai
import numpy as np
from gaze_estimation.detected_recognitions import DetectedRecognitions


class GazePositionEstimation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self,
              img_frames: dai.Node.Output,
              gaze_detection_data: dai.Node.Output,
              head_position_det: dai.Node.Output,
              head_position_depth: dai.Node.Output,
              head_rotation_nn_data: dai.Node.Output
              ):
        self.link_args(img_frames, gaze_detection_data, head_position_det, head_position_depth, head_rotation_nn_data)
        return self

    def process(self,
                img_frame: dai.ImgFrame,
                gaze_detection_data: dai.Buffer,
                head_position_det: dai.SpatialImgDetections,
                head_position_depth: dai.ImgFrame,
                head_rotation_nn_data: dai.NNData
                ):
        assert (isinstance(gaze_detection_data, DetectedRecognitions))
        frame = img_frame.getCvFrame()
        gaze_nn = gaze_detection_data.nn_data



        # TODO
        '''
        - rotate gaze vector based on head rotation angles
            - get gaze vector
            - get head rotation angles
            - create rotation matrix
            - apply rotation
        - normalize gaze vector
        - calculate scaling factor of the gaze vector
            - head_pos_z + t * gaze_z = 0 ( find t )
        - calculate intersection point of scaled and shifted gaze vector with the xy
            - scale the gaze vector by the found t and add up with the head vector
        
        '''