import cv2
import depthai as dai
import numpy as np

TARGET_SHAPE = (400, 640)
MODEL_MAX_DISP = 192
NN_DISP_MULTIPLIER = 255.0 / MODEL_MAX_DISP


class StereoMatching(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        disparity: dai.Node.Output,
        nn: dai.Node.Output,
        nn_shape: tuple[int, int],
        max_disparity: int,
    ) -> "StereoMatching":
        self.link_args(disparity, nn)
        self.sendProcessingToPipeline(True)
        self._nn_shape = nn_shape
        self._scale_multiplier = TARGET_SHAPE[1] / nn_shape[1]
        self._stereo_disp_multiplier = 255.0 / max_disparity
        return self

    def process(self, disparity: dai.ImgFrame, nn: dai.NNData) -> None:
        nn_output = np.array(nn.getData()).view(np.float16)
        frame = disparity.getFrame()

        nn_output = nn_output.reshape(
            (1, 2, self._nn_shape[0], self._nn_shape[1])
        ).squeeze()
        nn_disp = nn_output[0, :, :].astype("uint8")

        nn_disp = (nn_disp * NN_DISP_MULTIPLIER * self._scale_multiplier).astype(
            np.uint8
        )
        nn_disp = cv2.resize(nn_disp, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        nn_disp_vis = cv2.applyColorMap(nn_disp, cv2.COLORMAP_INFERNO)
        nn_disp_vis = cv2.putText(
            nn_disp_vis,
            "NN",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

        stereo_disp = (frame * self._stereo_disp_multiplier).astype(np.uint8)
        stereo_disp_vis = cv2.applyColorMap(stereo_disp, cv2.COLORMAP_INFERNO)
        stereo_disp_vis = cv2.putText(
            stereo_disp_vis,
            "Stereo Disp",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

        vis = np.concatenate([nn_disp_vis, stereo_disp_vis], axis=1)
        cv2.imshow("Disparity", vis)

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()
