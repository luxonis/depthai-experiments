from pathlib import Path
import depthai as dai
import cv2
import numpy as np

SHAPE = 300

class DisplayAll(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, rgb_out : dai.Node.Output, nn_blur : dai.Node.Output, nn_edge : dai.Node.Output,
            nn_diff : dai.Node.Output, nn_concat : dai.Node.Output) -> "DisplayAll":
        self.link_args(rgb_out, nn_blur, nn_edge, nn_diff, nn_concat)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, rgb_frame : dai.ImgFrame, nn_blur : dai.NNData, nn_edge : dai.NNData,
                nn_diff : dai.NNData, nn_concat : dai.NNData) -> None:
        cv2.imshow("Color", rgb_frame.getCvFrame())
        cv2.imshow("Blur", self.get_frame(nn_blur))
        cv2.imshow("Edge", self.get_frame(nn_edge))
        cv2.imshow("Diff", self.get_frame_diff(nn_diff, (720, 720)))
        cv2.imshow("Concat", self.get_frame_concat(nn_concat))

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def get_frame(self, imfFrame):
        shape = (3, SHAPE, SHAPE)
        return np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8)


    def get_frame_diff(self, data: dai.NNData, shape):
        diff = data.getFirstTensor().flatten().reshape(shape)
        colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)
    

    def get_frame_concat(self, nn_data):
        shape = (3, SHAPE, SHAPE * 3)
        inNn = np.array(nn_data.getData())
        frame = inNn.view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8).copy()
        return frame


with dai.Pipeline() as pipeline:

    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(SHAPE, SHAPE)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # BLUR
    nn_blur = pipeline.create(dai.node.NeuralNetwork)
    nn_blur.setBlobPath(Path(__file__).parent / "models/blur_simplified_openvino_2021.4_6shave.blob")

    camRgb.preview.link(nn_blur.input)

    # EDGE
    nn_edge = pipeline.create(dai.node.NeuralNetwork)
    nn_edge.setBlobPath(Path(__file__).parent / "models/edge_simplified_openvino_2021.4_6shave.blob")
    camRgb.preview.link(nn_edge.input)

    # DIFF
    img_manip = pipeline.create(dai.node.ImageManip)
    img_manip.initialConfig.setResize(720, 720) 
    img_manip.setMaxOutputFrameSize(720*720*3)

    camRgb.preview.link(img_manip.inputImage)

    nn_diff = pipeline.create(dai.node.NeuralNetwork)
    nn_diff.setBlobPath(Path(__file__).parent / "models/diff_openvino_2021.4_6shave.blob")

    script = pipeline.create(dai.node.Script)
    img_manip.out.link(script.inputs['in'])
    script.setScript("""
    old = node.io['in'].get()
    while True:
        frame = node.io['in'].get()
        node.io['img1'].send(old)
        node.io['img2'].send(frame)
        old = frame
    """)
    script.outputs['img1'].link(nn_diff.inputs['img1'])
    script.outputs['img2'].link(nn_diff.inputs['img2'])

    # CONCAT
    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipLeft = pipeline.create(dai.node.ImageManip)
    manipLeft.initialConfig.setResize(SHAPE, SHAPE)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipLeft.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    left.out.link(manipLeft.inputImage)

    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipRight = pipeline.create(dai.node.ImageManip)
    manipRight.initialConfig.setResize(SHAPE, SHAPE)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipRight.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    right.out.link(manipRight.inputImage)

    nn_concat = pipeline.create(dai.node.NeuralNetwork)
    nn_concat.setBlobPath(Path(__file__).parent / "models/concat_openvino_2021.4_6shave.blob")
    nn_concat.setNumInferenceThreads(2)

    manipLeft.out.link(nn_concat.inputs['img1'])
    camRgb.preview.link(nn_concat.inputs['img2'])
    manipRight.out.link(nn_concat.inputs['img3'])

    pipeline.create(DisplayAll).build(
        rgb_out=camRgb.preview,
        nn_blur=nn_blur.out,
        nn_edge=nn_edge.out,
        nn_diff=nn_diff.out,
        nn_concat=nn_concat.out
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
