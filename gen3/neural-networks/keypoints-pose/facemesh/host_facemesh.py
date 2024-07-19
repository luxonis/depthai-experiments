import numpy as np
import cv2
import depthai as dai
from effect import EffectRenderer2D

# The facemesh model is trained on the whole head, not just on the face
FACE_BBOX_PADDING = 0.06

def generate_colors_from_z(z_values):
    minZ = min(z_values)
    maxZ = max(z_values)
    normalized_z_values = (z_values - minZ) / (maxZ - minZ)
    return [(255 - int((1 - value) * 255), 0, 255 - int(value * 255)) for value in normalized_z_values]

class Facemesh(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.effect_renderer = EffectRenderer2D("facepaint.png")

    def build(self, full: dai.Node.Output, preview: dai.Node.Output
            , face_nn: dai.Node.Output, landmarks_nn: dai.Node.Output) -> "Facemesh":
        self.link_args(full, preview, face_nn, landmarks_nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, full: dai.ImgFrame, preview: dai.ImgFrame, face_nn: dai.ImgDetections, landmarks_nn: dai.NNData):
        preview_both = full.getCvFrame()
        preview_effect = preview.getCvFrame()
        preview_landmarks = preview.getCvFrame()
        x_shift = int((preview_both.shape[1] - preview_both.shape[0]) / 2)

        self.displayDetections(preview_both, face_nn.detections, x_shift)
        cv2.rectangle(preview_both, (int((preview_both.shape[1] - preview_both.shape[0])/2), 0)
                      , (int((preview_both.shape[0] + preview_both.shape[1])/2), preview_both.shape[0]), (0, 0, 255), 2)

        threshhold = landmarks_nn.getTensor('conv2d_31').reshape((1,))
        threshhold = 1 / (1 + np.exp(-threshhold[0]))  # sigmoid on threshhold
        landmarks = landmarks_nn.getTensor('conv2d_21').reshape((468, 3))

        if threshhold > 0.5 and len(face_nn.detections) != 0:
            det = face_nn.detections[0] # Take first
            colors = generate_colors_from_z(np.array([ldm[2] for ldm in landmarks]))
            scale_to_full = preview_both.shape[0] / preview_landmarks.shape[0]

            xmin = (det.xmin - FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            ymin = (det.ymin - FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            width = (det.xmax - det.xmin + 2 * FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            height = (det.ymax - det.ymin + 2 * FACE_BBOX_PADDING) * preview_landmarks.shape[0]
            # The input shape of the model is 192x192
            landmarks *= np.array([width / 192, height / 192, 1])

            landmarks_full = landmarks.copy()
            xmin_full = xmin * scale_to_full
            ymin_full = ymin * scale_to_full
            landmarks_full *= np.array([scale_to_full, scale_to_full, 1])

            for ldm, ldm_full, col in zip(landmarks, landmarks_full, colors):
                cv2.circle(preview_landmarks, (int(xmin + ldm[0]), int(ymin + ldm[1])), 1, col, 1)
                cv2.circle(preview_both, (int(x_shift + xmin_full + ldm_full[0]), int(ymin_full + ldm_full[1])), 1, col, 1)
            preview_effect = self.effect_renderer.render_effect(preview_effect, landmarks, int(xmin), int(ymin))
            preview_both = self.effect_renderer.render_effect(preview_both, landmarks_full, int(x_shift + xmin_full), int(ymin_full))

        combined = np.concatenate((preview_landmarks, preview_effect), axis=1)
        cv2.imshow("Preview", combined)
        cv2.imshow("Full", preview_both)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def displayDetections(self, frame: np.ndarray, detections: list, x_shift: int):
        for detection in detections:
            bbox = (np.array((detection.xmin, detection.ymin, detection.xmax, detection.ymax)) * frame.shape[0]).astype(int)
            bbox_padded = (np.array((detection.xmin - FACE_BBOX_PADDING, detection.ymin - FACE_BBOX_PADDING
                                     , detection.xmax + FACE_BBOX_PADDING, detection.ymax + FACE_BBOX_PADDING)) * frame.shape[0]).astype(int)
            cv2.rectangle(frame, (x_shift + bbox[0], bbox[1]), (x_shift + bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (x_shift + bbox_padded[0], bbox_padded[1]), (x_shift + bbox_padded[2], bbox_padded[3]), (0, 0, 255), 2)
