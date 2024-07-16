import numpy as np
import cv2
import depthai as dai
import os

SHAPE = 300

class DisplayConcat(dai.node.HostNode):
    def __init__(self):
        self.shape = (3, SHAPE, SHAPE * 3)
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

    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipLeft = pipeline.create(dai.node.ImageManip)
    manipLeft.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipLeft.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    left.out.link(manipLeft.inputImage)

    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipRight = pipeline.create(dai.node.ImageManip)
    manipRight.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipRight.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    right.out.link(manipRight.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    # to test from vs code
    # dirname = os.path.dirname(__file__)
    # path = os.path.join(dirname, "models", "concat_openvino_2021.4_6shave.blob")
    # nn.setBlobPath(path)
    nn.setBlobPath("models/concat_openvino_2021.4_6shave.blob")
    nn.setNumInferenceThreads(2)

    manipLeft.out.link(nn.inputs['img1'])
    camRgb.preview.link(nn.inputs['img2'])
    manipRight.out.link(nn.inputs['img3'])

    pipeline.create(DisplayConcat).build(
        nn.out
    )

    pipeline.run()


# p = dai.Pipeline()
# p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# camRgb = p.create(dai.node.ColorCamera)
# camRgb.setPreviewSize(SHAPE, SHAPE)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# left = p.create(dai.node.MonoCamera)
# left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
# manipLeft = p.create(dai.node.ImageManip)
# manipLeft.initialConfig.setResize(300, 300)
# # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
# manipLeft.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
# left.out.link(manipLeft.inputImage)

# right = p.create(dai.node.MonoCamera)
# right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
# manipRight = p.create(dai.node.ImageManip)
# manipRight.initialConfig.setResize(300, 300)
# # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
# manipRight.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
# right.out.link(manipRight.inputImage)

# NN that concatenates inputs
# nn = p.create(dai.node.NeuralNetwork)
# nn.setBlobPath("models/concat_openvino_2021.4_6shave.blob")
# nn.setNumInferenceThreads(2)

# manipLeft.out.link(nn.inputs['img1'])
# camRgb.preview.link(nn.inputs['img2'])
# manipRight.out.link(nn.inputs['img3'])

# Send bouding box from the NN to the host via XLink
# nn_xout = p.create(dai.node.XLinkOut)
# nn_xout.setStreamName("nn")
# nn.out.link(nn_xout.input)

# Pipeline is defined, now we can connect to the device
# with dai.Device(p) as device:
#     qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
#     shape = (3, SHAPE, SHAPE * 3)

#     while True:
#         inNn = np.array(qNn.get().getData())
#         frame = inNn.view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8).copy()

#         cv2.imshow("Concat", frame)

#         if cv2.waitKey(1) == ord('q'):
#             break
