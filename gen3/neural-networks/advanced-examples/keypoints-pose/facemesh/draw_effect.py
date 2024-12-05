import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import Keypoints
from effect import EffectRenderer2D


class DrawEffect(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.effect_renderer = EffectRenderer2D("facepaint.png")
        self.output_mask = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.output_landmarks = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self, preview: dai.Node.Output, landmarks_nn: dai.Node.Output
    ) -> "DrawEffect":
        self.link_args(preview, landmarks_nn)
        return self

    def process(self, preview: dai.ImgFrame, landmark_keypoints: dai.Buffer):
        preview_effect = preview.getCvFrame()
        assert isinstance(landmark_keypoints, Keypoints)

        landmarks = np.array(
            [
                [keypoint.x, keypoint.y, keypoint.z]
                for keypoint in landmark_keypoints.keypoints
            ]
        )
        if len(landmarks) > 0:
            preview_effect = self.effect_renderer.render_effect(
                preview_effect, landmarks
            )

        output_mask = dai.ImgFrame()
        output_mask.setCvFrame(preview_effect, dai.ImgFrame.Type.BGR888p)
        output_mask.setTimestamp(preview.getTimestamp())
        output_mask.setSequenceNum(preview.getSequenceNum())
        self.output_mask.send(output_mask)
