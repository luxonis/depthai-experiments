import numpy as np
import cv2
import depthai as dai
from effect import EffectRenderer2D
from depthai_nodes.ml.messages import Keypoints, ImgDetectionsExtended


# The facemesh model is trained on the whole head, not just on the face



class Facemesh(dai.node.HostNode):
    FACE_BBOX_PADDING = 0.1


    def __init__(self) -> None:
        super().__init__()
        self.effect_renderer = EffectRenderer2D("facepaint.png")
        self.output_mask = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self.output_landmarks = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        

    def build(self, preview: dai.Node.Output
            , face_nn: dai.Node.Output, landmarks_nn: dai.Node.Output) -> "Facemesh":
        self.link_args(preview, face_nn, landmarks_nn)
        return self


    def process(self, preview: dai.ImgFrame, face_nn: dai.Buffer, landmark_keypoints: dai.Buffer):
        preview_effect = preview.getCvFrame()
        preview_landmarks = preview_effect.copy()

        assert(isinstance(face_nn, ImgDetectionsExtended))
        if len(face_nn.detections) != 0:
            det = face_nn.detections[0] # Take first

            assert(isinstance(landmark_keypoints, Keypoints))
            landmarks = np.array([[keypoint.x, keypoint.y, keypoint.z] for keypoint in landmark_keypoints.keypoints])

            xmin = (det.xmin - self.FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            ymin = (det.ymin - self.FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            width = (det.xmax - det.xmin + 2 * self.FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            height = (det.ymax - det.ymin + 2 * self.FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            landmarks *= np.array([192, 192, 255])
            landmarks *= np.array([width / 192, height / 192, 1])

            colors = self.generate_colors_from_z(np.array([ldm[2] for ldm in landmarks]))

            # The input shape of the model is 192x192

            for ldm, col in zip(landmarks, colors):
                cv2.circle(preview_landmarks, (int(xmin + ldm[0]), int(ymin + ldm[1])), 1, col, 1)
            preview_effect = self.effect_renderer.render_effect(preview_effect, landmarks, int(xmin), int(ymin))

        output_landmarks = dai.ImgFrame()
        output_landmarks.setCvFrame(preview_landmarks, dai.ImgFrame.Type.BGR888i)
        self.output_landmarks.send(output_landmarks)

        output_mask = dai.ImgFrame()
        output_mask.setCvFrame(preview_effect, dai.ImgFrame.Type.BGR888i)
        self.output_mask.send(output_mask)


    def generate_colors_from_z(self, z_values):
        minZ = min(z_values)
        maxZ = max(z_values)
        normalized_z_values = (z_values - minZ) / (maxZ - minZ)
        return [(255 - int((1 - value) * 255), 0, 255 - int(value * 255)) for value in normalized_z_values]


    def displayDetections(self, frame: np.ndarray, detections: list[dai.ImgDetection], x_shift: int):
        for detection in detections:
            bbox = (np.array((detection.xmin, detection.ymin, detection.xmax, detection.ymax)) * frame.shape[0]).astype(int)
            bbox_padded = (np.array((detection.xmin - self.FACE_BBOX_PADDING, detection.ymin - self.FACE_BBOX_PADDING
                                     , detection.xmax + self.FACE_BBOX_PADDING, detection.ymax + self.FACE_BBOX_PADDING)) * frame.shape[0]).astype(int)
            cv2.rectangle(frame, (x_shift + bbox[0], bbox[1]), (x_shift + bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (x_shift + bbox_padded[0], bbox_padded[1]), (x_shift + bbox_padded[2], bbox_padded[3]), (0, 0, 255), 2)
