from math import cos, sin
import numpy as np
import cv2
import depthai as dai

from utils import frame_norm, copy_timestamps
from numpy_buffer import NumpyBuffer


class GazeEstimation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._debug = True
        self._laser = False
        self._camera = True
        self.output = dai.Node.Output(self, possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    
    def build(self, 
              img_frames: dai.Node.Output, 
              gaze_nn: dai.Node.Output, 
              right_bbox_buffers: NumpyBuffer, 
              left_bbox_buffers: NumpyBuffer, 
              bbox_buffers: NumpyBuffer, 
              nose_buffers: dai.Node.Output, 
              pose_buffers: dai.Node.Output
        ) -> "GazeEstimation":
        self.link_args(img_frames, gaze_nn, right_bbox_buffers, left_bbox_buffers, bbox_buffers, nose_buffers, pose_buffers)
        self.sendProcessingToPipeline(True)
        return self
    

    def set_debug(self, debug: bool) -> None:
        self._debug = debug
    

    def set_laser(self, laser: bool) -> None:
        self._laser = laser


    def set_camera(self, camera: bool) -> None:
        self._camera = camera


    def process(self, 
                img_frame: dai.ImgFrame, 
                gaze_nn: dai.NNData, 
                right_bbox_buffer: dai.Buffer,
                left_bbox_buffer: dai.Buffer,
                bboxes_buffer: dai.Buffer,
                nose_buffer: dai.Buffer,
                pose_buffer: dai.Buffer,
        ) -> None:
        assert(isinstance(right_bbox_buffer, NumpyBuffer))
        assert(isinstance(left_bbox_buffer, NumpyBuffer))
        assert(isinstance(bboxes_buffer, NumpyBuffer))
        assert(isinstance(nose_buffer, NumpyBuffer))
        assert(isinstance(pose_buffer, NumpyBuffer))
        frame: np.ndarray = img_frame.getCvFrame()
        debug_frame = frame.copy()

        gaze_layers = gaze_nn.getAllLayerNames()
        if not gaze_layers:
            self.output.send(img_frame)
            return
        
        batches = gaze_nn.getFirstTensor().shape[0]
        left_bbox = left_bbox_buffer.getData()
        right_bbox = right_bbox_buffer.getData()
        gaze = gaze_nn.getFirstTensor()
        bboxes = bboxes_buffer.getData()
        nose = nose_buffer.getData()
        pose = pose_buffer.getData()
        if self._debug:  # face
            for batch_index in range(batches):
                batch_left_bbox = left_bbox[batch_index]
                batch_right_bbox = right_bbox[batch_index]
                batch_gaze = gaze[batch_index]
                if batch_gaze is not None and batch_left_bbox is not None and batch_right_bbox is not None:
                    re_x = (batch_right_bbox[0] + batch_right_bbox[2]) // 2
                    re_y = (batch_right_bbox[1] + batch_right_bbox[3]) // 2
                    le_x = (batch_left_bbox[0] + batch_left_bbox[2]) // 2
                    le_y = (batch_left_bbox[1] + batch_left_bbox[3]) // 2

                    x, y = (batch_gaze * 100).astype(int)[:2]

                    if self._laser:
                        beam_img = np.zeros(debug_frame.shape, np.uint8)
                        for t in range(10)[::-2]:
                            cv2.line(beam_img, (re_x, re_y), ((re_x + x*100), (re_y - y*100)), (0, 0, 255-t*10), t*2)
                            cv2.line(beam_img, (le_x, le_y), ((le_x + x*100), (le_y - y*100)), (0, 0, 255-t*10), t*2)
                        debug_frame |= beam_img

                    else:
                        cv2.arrowedLine(debug_frame, (le_x, le_y), (le_x + x, le_y - y), (255, 0, 255), 3)
                        cv2.arrowedLine(debug_frame, (re_x, re_y), (re_x + x, re_y - y), (255, 0, 255), 3)

                if not self._laser:
                    batch_bboxes = bboxes[batch_index]
                    bbox = frame_norm(frame, batch_bboxes)
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                    batch_nose = nose[batch_index]
                    if batch_nose is not None:
                        cv2.circle(debug_frame, (batch_nose[0], batch_nose[1]), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
                    if batch_left_bbox is not None:
                        cv2.rectangle(debug_frame, (batch_left_bbox[0], batch_left_bbox[1]), (batch_left_bbox[2], batch_left_bbox[3]), (245, 10, 10), 2)
                    if batch_right_bbox is not None:
                        cv2.rectangle(debug_frame, (batch_right_bbox[0], batch_right_bbox[1]), (batch_right_bbox[2], batch_right_bbox[3]), (245, 10, 10), 2)
                    batch_pose = pose[batch_index]
                    if batch_pose is not None and batch_nose is not None:
                        self._draw_3d_axis(debug_frame, batch_pose, batch_nose)

        output_frame = dai.ImgFrame()
        copy_timestamps(img_frame, output_frame)
        output_frame.setType(dai.ImgFrame.Type.BGR888i)
        output_frame.setSize(img_frame.getWidth(), img_frame.getHeight())
        output_frame.setFrame(debug_frame)
        self.output.send(output_frame)


    def _draw_3d_axis(self, image, head_pose, origin, size=50):
        # From https://github.com/openvinotoolkit/open_model_zoo/blob/b1ff98b64a6222cf6b5f3838dc0271422250de95/demos/gaze_estimation_demo/cpp/src/results_marker.cpp#L50
        origin_x,origin_y = origin
        yaw,pitch, roll = np.array(head_pose)*np.pi / 180

        sinY = sin(yaw)
        sinP = sin(pitch)
        sinR = sin(roll)

        cosY = cos(yaw)
        cosP = cos(pitch)
        cosR = cos(roll)
        # X axis (red)
        x1 = origin_x + size * (cosR * cosY + sinY * sinP * sinR)
        y1 = origin_y + size * cosP * sinR
        cv2.line(image, (origin_x, origin_y), (int(x1), int(y1)), (0, 0, 255), 3)

        # Y axis (green)
        x2 = origin_x + size * (cosR * sinY * sinP + cosY * sinR)
        y2 = origin_y - size * cosP * cosR
        cv2.line(image, (origin_x, origin_y), (int(x2), int(y2)), (0, 255, 0), 3)

        # Z axis (blue)
        x3 = origin_x + size * (sinY * cosP)
        y3 = origin_y + size * sinP
        cv2.line(image, (origin_x, origin_y), (int(x3), int(y3)), (255, 0, 0), 2)

        return image