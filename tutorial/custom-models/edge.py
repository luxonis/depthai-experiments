import depthai as dai

from host_nodes.host_edge import ReshapeNNOutputEdge, EDGE_SHAPE
from host_nodes.host_display import Display

with dai.Pipeline() as pipeline:
    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=(EDGE_SHAPE, EDGE_SHAPE), type=dai.ImgFrame.Type.BGR888p)

    # NN that detects edges in the image
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/edge_simplified_openvino_2021.4_6shave.blob")
    preview.link(nn.input)

    reshape = pipeline.create(ReshapeNNOutputEdge).build( 
        nn_out=nn.out
    )
    
    color = pipeline.create(Display).build(frame=preview)
    color.setName("Color")

    edge = pipeline.create(Display).build(frame=reshape.output)
    edge.setName("Edge")

    pipeline.run()