import numpy as np
import cv2
import depthai as dai

SHAPE = 300


class DisplayBlur(dai.node.HostNode):
    def __init__(self):
        self.shape = (3, SHAPE, SHAPE)
        super().__init__()

    def build(self, rgb_out : dai.Node.Output, nn_out : dai.Node.Output) -> "DisplayBlur":
        self.link_args(rgb_out, nn_out)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, rgb_frame : dai.ImgFrame, nn_det : dai.NNData) -> None:
        cv2.imshow("Blur", self.get_frame(nn_det))
        cv2.imshow("Color", rgb_frame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
        pass

    def get_frame(self, imfFrame):
        return np.array(imfFrame.getData()).view(np.float16).reshape(self.shape).transpose(1, 2, 0).astype(np.uint8)


with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)
    
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    camRgb.setPreviewSize(SHAPE, SHAPE)
    camRgb.setInterleaved(False)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/blur_simplified_openvino_2021.4_6shave.blob")

    camRgb.preview.link(nn.input)

    pipeline.create(DisplayBlur).build(
        camRgb.preview,
        nn.out
    )

    pipeline.run()
