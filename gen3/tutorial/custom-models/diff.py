import os
import numpy as np
import cv2
import depthai as dai

class DisplayDiff(dai.node.HostNode):
    def __init__(self):
        super().__init__()
    
    def build(self, colorOut : dai.Node.Output, nnOut : dai.Node.Output) -> "DisplayDiff":
        self.link_args(colorOut, nnOut)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, camColor : dai.ImgFrame, nnData : dai.NNData) -> None:
        cv2.imshow("Diff", self.get_frame(nnData, (720, 720)))
        cv2.imshow("Color", camColor.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
        

    def get_frame(data: dai.NNData, shape):
        diff = np.array(data.getFirstLayerFp16()).reshape(shape)
        colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)


with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setVideoSize(720, 720)
    camRgb.setPreviewSize(720, 720)
    camRgb.setInterleaved(False)

    # NN that detects faces in the image
    nn = pipeline.create(dai.node.NeuralNetwork)
    # testing from vs code
    # dirname = os.path.dirname(__file__)
    # path = os.path.join(dirname, "models", "diff_openvino_2021.4_6shave.blob")
    # nn.setBlobPath(path)
    nn.setBlobPath("models/diff_openvino_2021.4_6shave.blob")

    script = pipeline.create(dai.node.Script)
    camRgb.preview.link(script.inputs['in'])
    script.setScript("""
    old = node.io['in'].get()
    while True:
        frame = node.io['in'].get()
        node.io['img1'].send(old)
        node.io['img2'].send(frame)
        old = frame
    """)
    script.outputs['img1'].link(nn.inputs['img1'])
    script.outputs['img2'].link(nn.inputs['img2'])


    pipeline.create(DisplayDiff).build(
        camRgb.preview,
        nn.out
    )

    pipeline.run()


# # Pipeline is defined, now we can connect to the device
# with dai.Device(p) as device:
#     qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
#     qCam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

#     def get_frame(data: dai.NNData, shape):
#         diff = np.array(data.getFirstLayerFp16()).reshape(shape)
#         colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
#         return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)    

#     while True:
#         cv2.imshow("Diff", get_frame(qNn.get(), (720, 720)))
#         cv2.imshow("Color", qCam.get().getCvFrame())

#         if cv2.waitKey(1) == ord('q'):
#             break
