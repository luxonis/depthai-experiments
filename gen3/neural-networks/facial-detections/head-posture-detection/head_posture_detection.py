import cv2
import depthai as dai
import numpy as np

from detected_recognitions import DetectedRecognitions


class HeadPostureDetection(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._min_threshold = 15
        self._bg_color = (0, 0, 0)
        self._color = (255, 255, 255)
        self._text_type = cv2.FONT_HERSHEY_SIMPLEX
        self._line_type = cv2.LINE_AA


    def build(self, img_frames: dai.Node.Output, detected_recognitions: dai.Node.Output) -> "HeadPostureDetection":
        self.link_args(img_frames, detected_recognitions)
        return self
    

    def set_min_threshold(self, min_threshold: float) -> None:
        self._min_threshold = min_threshold

    
    def set_bg_color(self, bg_color: tuple[int, int, int]) -> None:
        self._bg_color = bg_color

    
    def set_color(self, color: tuple[int, int, int]) -> None:
        self._color = color
    

    def set_text_type(self, text_type: int) -> None:
        self._text_type = text_type

    
    def set_line_type(self, line_type: int) -> None:
        self._line_type = line_type
        

    def process(self, img_frame: dai.ImgFrame, detected_recognitions: dai.Buffer) -> None:
        frame = img_frame.getCvFrame()
        assert(isinstance(detected_recognitions, DetectedRecognitions))
        detections = detected_recognitions.img_detections.detections

        for i, detection in enumerate(detections):
            bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            rec = detected_recognitions.nn_data[i]
            yaw = rec.getTensor('angle_y_fc').flatten()[0]
            pitch = rec.getTensor('angle_p_fc').flatten()[0]
            roll = rec.getTensor('angle_r_fc').flatten()[0]
            self._decode_pose(yaw, pitch, roll, bbox, frame)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
            y = (bbox[1] + bbox[3]) // 2
            if isinstance(detection, dai.SpatialImgDetection):
                coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _frame_norm(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    

    def _decode_pose(self, yaw, pitch, roll, face_bbox, frame):
        """
        pitch > 0 Head down, < 0 look up
        yaw > 0 Turn right < 0 Turn left
        roll > 0 Tilt right, < 0 Tilt left
        """
        self._putText(frame,"pitch:{:.0f}, yaw:{:.0f}, roll:{:.0f}".format(pitch,yaw,roll),(face_bbox[0]+10-15,face_bbox[1]-15))

        vals = np.array([abs(pitch),abs(yaw),abs(roll)])
        max_index = np.argmax(vals)

        if vals[max_index] < self._min_threshold: return

        if max_index == 0:
            if pitch > 0: txt = "Look down"
            else: txt = "Look up"
        elif max_index == 1:
            if yaw > 0: txt = "Turn right"
            else: txt = "Turn left"
        elif max_index == 2:
            if roll > 0: txt = "Tilt right"
            else: txt = "Tilt left"
        self._putText(frame,txt, (face_bbox[0]+10,face_bbox[1]+30), size=1, bold=True)


    def _putText(self, frame: np.ndarray, text: str, coords: tuple[int, int], size=0.6, bold=False) -> None:
        mult = 2 if bold else 1
        cv2.putText(frame, text, coords, self._text_type, size, self._bg_color, 3*mult, self._line_type)
        cv2.putText(frame, text, coords, self._text_type, size, self._color, 1*mult, self._line_type)
