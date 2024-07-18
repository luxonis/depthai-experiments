import numpy as np
import cv2
import depthai as dai

SHAPE = 300

class DisplayEdges(dai.node.HostNode):
    def __init__(self):
        self.shape = (3, SHAPE, SHAPE)
        super().__init__()

    def build(self, rgbOut : dai.Node.Output, nnOut : dai.Node.Output) -> "DisplayEdges":
        self.link_args(rgbOut, nnOut)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, rgbFrame : dai.ImgFrame, nnFrame : dai.NNData) -> None:
        cv2.imshow("Laplacian edge detection", self.get_frame(nnFrame))
        cv2.imshow("Color", rgbFrame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
    
    def get_frame(self, imfFrame):
        return np.array(imfFrame.getData()).view(np.float16).reshape(self.shape).transpose(1, 2, 0).astype(np.uint8)

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(SHAPE, SHAPE)
    camRgb.setInterleaved(False)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)

    # NN that detects edges in the image
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/edge_simplified_openvino_2021.4_6shave.blob")
    camRgb.preview.link(nn.input)

    pipeline.create(DisplayEdges).build(
        camRgb.preview,
        nn.out
    )

    pipeline.run()
