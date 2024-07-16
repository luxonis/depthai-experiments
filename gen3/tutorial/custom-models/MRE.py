import numpy as np
import cv2
import depthai as dai

SHAPE = 300

class DisplayConcat(dai.node.HostNode):
    def __init__(self):
        self.shape = (3, SHAPE, SHAPE)
        super().__init__()

    def build(self, nn_out : dai.Node.Output) -> "DisplayConcat":
        self.link_args(nn_out)
        self.sendProcessingToPipeline(True)
        return self # crashes after this line, probably problem with self.link_args(nn_out)
    
    def process(self, nn_data : dai.NNData): # program doesn't get here
        inNn = np.array(nn_data.getData())
        frame = inNn.view(np.float16).reshape(self.shape).transpose(1, 2, 0).astype(np.uint8).copy()

        cv2.imshow("Concat", frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(SHAPE, SHAPE)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)

    nn = pipeline.create(dai.node.NeuralNetwork)
    # to test from vs code
    # dirname = os.path.dirname(__file__)
    # path = os.path.join(dirname, "models", "concat_openvino_2021.4_6shave.blob")
    # nn.setBlobPath(path)
    nn.setBlobPath("models/concat_openvino_2021.4_6shave.blob")
    nn.setNumInferenceThreads(2)

    camRgb.preview.link(nn.inputs['img1'])

    pipeline.create(DisplayConcat).build(
        nn.out
    )

    pipeline.run()


