import depthai as dai

from host_nodes.host_display import Display
from host_nodes.host_blur import ReshapeNNOutputBlur, BLUR_SHAPE


with dai.Pipeline() as pipeline:
    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=(BLUR_SHAPE, BLUR_SHAPE), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/blur_simplified_openvino_2021.4_6shave.blob")

    preview.link(nn.input)

    reshape = pipeline.create(ReshapeNNOutputBlur).build(
        nn_out=nn.out
    )
    
    color_display = pipeline.create(Display).build(preview)
    color_display.setName("Color Display")

    nn_display = pipeline.create(Display).build(reshape.output)
    nn_display.setName("Blur Display")
    
    pipeline.run()