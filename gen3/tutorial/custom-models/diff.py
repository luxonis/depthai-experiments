import depthai as dai

from host_nodes.host_diff import ColorizeDiff, DIFF_SHAPE
from host_nodes.host_display import Display


with dai.Pipeline() as pipeline:
    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=DIFF_SHAPE, type=dai.ImgFrame.Type.BGR888p)

    # NN that detects faces in the image
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/diff_openvino_2021.4_6shave.blob")

    script = pipeline.create(dai.node.Script)
    preview.link(script.inputs['in'])
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

    colorized = pipeline.create(ColorizeDiff).build(
        nnOut=nn.out
    )
    
    diff = pipeline.create(Display).build(frame=colorized.output)
    diff.setName("Diff")
    rgb = pipeline.create(Display).build(frame=preview)
    rgb.setName("Color")

    pipeline.run()