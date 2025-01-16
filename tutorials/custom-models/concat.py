import depthai as dai

from host_nodes.host_concat import ReshapeNNOutputConcat, CONCAT_SHAPE
from host_nodes.host_display import Display


with dai.Pipeline() as pipeline:
    cam_rgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    preview = cam_rgb.requestOutput(
        size=(CONCAT_SHAPE, CONCAT_SHAPE), type=dai.ImgFrame.Type.BGR888p
    )

    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipLeft = pipeline.create(dai.node.ImageManip)
    manipLeft.initialConfig.setResize(CONCAT_SHAPE, CONCAT_SHAPE)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipLeft.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    left.out.link(manipLeft.inputImage)

    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    manipRight = pipeline.create(dai.node.ImageManip)
    manipRight.initialConfig.setResize(CONCAT_SHAPE, CONCAT_SHAPE)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipRight.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    right.out.link(manipRight.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/concat_openvino_2021.4_6shave.blob")
    nn.setNumInferenceThreads(2)

    manipLeft.out.link(nn.inputs["img1"])
    preview.link(nn.inputs["img2"])
    manipRight.out.link(nn.inputs["img3"])

    reshape = pipeline.create(ReshapeNNOutputConcat).build(nn_out=nn.out)

    concat = pipeline.create(Display).build(frame=reshape.output)
    concat.setName("Concat")

    pipeline.run()
