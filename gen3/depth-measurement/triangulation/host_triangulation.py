import cv2
import math
import depthai as dai
import numpy as np

class StereoInference:
    def __init__(self, device: dai.Device, resolution_num, width, heigth) -> None:
        calibData = device.readCalibration()
        baseline = calibData.getBaselineDistance(useSpecTranslation=True) * 10  # mm

        # Original mono frames shape
        self.original_heigth = resolution_num
        self.original_width = 640 if resolution_num == 400 else 1280
        self.hfov = calibData.getFov(dai.CameraBoardSocket.CAM_C)

        focalLength = self.get_focal_length_pixels(self.original_width, self.hfov)
        self.dispScaleFactor = baseline * focalLength

        # Cropped frame shape
        self.mono_width = width
        self.mono_heigth = heigth
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from
        # 720x720 (by ImageManip), so we need to multiple coords by 2.4 to get the correct disparity.
        self.resize_factor = self.original_heigth / self.mono_heigth

    def get_focal_length_pixels(self, pixel_width, hfov):
        return pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi/180)

    def calculate_depth(self, disparity_pixels: float):
        try:
            return self.dispScaleFactor / disparity_pixels
        except ZeroDivisionError:
            return 0

    def calculate_distance(self, c1, c2):
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from 720x720 (by ImageManip),
        # so we need to multiple coords by 2.4 (if using 720P resolution) to get the correct disparity.
        c1 = np.array(c1) * self.resize_factor
        c2 = np.array(c2) * self.resize_factor

        x_delta = c1[0] - c2[0]
        y_delta = c1[1] - c2[1]
        return math.sqrt(x_delta ** 2 + y_delta ** 2)

    def calc_angle(self, offset):
            return math.atan(math.tan(self.hfov / 2.0) * offset / (self.original_width / 2.0))

    def calc_spatials(self, coords, depth):
        x, y = coords
        bb_x_pos = x - self.mono_width / 2
        bb_y_pos = y - self.mono_heigth / 2

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = depth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)
        return [x,y,z]
class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)

class Triangulation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._leftColor = (255, 0, 0)
        self._rightColor = (0, 255, 0)
        self._textHelper = TextHelper()

    def build(self, face_left: dai.Node.Output, face_right: dai.Node.Output,
              face_nn_left: dai.Node.Output, face_nn_right : dai.Node.Output,
              landmarks_nn_left: dai.Node.Output, landmarks_nn_right: dai.Node.Output,
              device: dai.Device, resolution_number: int) -> "Triangulation":

        self.link_args(face_left, face_right, face_nn_left, face_nn_right, landmarks_nn_left, landmarks_nn_right)
        self.sendProcessingToPipeline(True)
        self._stereoInference = StereoInference(device, resolution_number, width=300, heigth=300)

        return self

    def process(self, face_left: dai.Buffer, face_right: dai.Buffer,
                nn_face_left: dai.Buffer, nn_face_right: dai.Buffer,
                nn_landmarks_left: dai.NNData, nn_landmarks_right: dai.NNData) -> None:

        frame_left, landmarks_left = self.process_side(face_left, nn_face_left, nn_landmarks_left, True)
        frame_right, landmarks_right = self.process_side(face_right, nn_face_right, nn_landmarks_right, False)

        combined = cv2.addWeighted(frame_left, 0.5, frame_right, 0.5, 0)

        if landmarks_left and landmarks_right:
            spatials = []
            for i in range(5):
                coords_left = landmarks_left[i]
                coords_right = landmarks_right[i]

                cv2.circle(frame_left, coords_left, 3, self._leftColor)
                cv2.circle(frame_right, coords_right, 3, self._rightColor)
                cv2.circle(combined, coords_left, 3, self._leftColor)
                cv2.circle(combined, coords_right, 3, self._rightColor)

                # Visualize disparity line frame
                cv2.line(combined, coords_right, coords_left, (0, 0, 255), 1)

                disparity = self._stereoInference.calculate_distance(coords_left, coords_right)
                depth = self._stereoInference.calculate_depth(disparity)
                spatial = self._stereoInference.calc_spatials(coords_right, depth)
                spatials.append(spatial)

                if i == 0:
                    y = 0
                    y_delta = 18
                    strings = [
                        "Disparity: {:.0f} pixels".format(disparity),
                        "X: {:.2f} m".format(spatial[0]/1000),
                        "Y: {:.2f} m".format(spatial[1]/1000),
                        "Z: {:.2f} m".format(spatial[2]/1000),
                    ]
                    for s in strings:
                        y += y_delta
                        self._textHelper.putText(combined, s, (10, y))

        cv2.imshow("Combined frame", np.concatenate((frame_left, combined, frame_right), axis=1))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def process_side(self, frame_face: dai.ImgFrame, nn_face: dai.ImgDetections
                     , nn_landmarks: dai.NNData, left: bool) -> tuple[np.ndarray, list[(int, int)]]:
        frame = frame_face.getCvFrame()
        color = self._leftColor if left else self._rightColor
        displayDetections(frame, nn_face.detections)

        landmarks_xy = []
        if len(nn_face.detections) != 0:
            det = nn_face.detections[0]  # Take first

            width = det.xmax - det.xmin
            height = det.ymax - det.ymin

            landmarks = nn_landmarks.getFirstTensor().astype(np.float16).reshape(5, 2)

            landmarks_xy = []
            for landmark in landmarks:
                x = int((landmark[0] * width + det.xmin) * 300)
                y = int((landmark[1] * height + det.ymin) * 300)
                landmarks_xy.append((x, y))

        return (frame, landmarks_xy)

def displayDetections(frame: np.ndarray, detections: list) -> None:
    for detection in detections:
        bbox = (np.array((detection.xmin, detection.ymin, detection.xmax, detection.ymax)) * frame.shape[0]).astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
