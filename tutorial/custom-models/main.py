from pathlib import Path
import depthai as dai

from host_nodes.host_display import Display
from host_nodes.host_blur import ReshapeNNOutputBlur
from host_nodes.host_edge import ReshapeNNOutputEdge
from host_nodes.host_diff import ColorizeDiff
from host_nodes.host_concat import ReshapeNNOutputConcat

SHAPE = 300

with dai.Pipeline() as pipeline:

    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=(SHAPE, SHAPE), type=dai.ImgFrame.Type.BGR888p)

    # BLUR
    nn_blur = pipeline.create(dai.node.NeuralNetwork)
    nn_blur.setBlobPath(Path(__file__).parent / "models/blur_simplified_openvino_2021.4_6shave.blob")

    preview.link(nn_blur.input)

    # EDGE
    nn_edge = pipeline.create(dai.node.NeuralNetwork)
    nn_edge.setBlobPath(Path(__file__).parent / "models/edge_simplified_openvino_2021.4_6shave.blob")
    preview.link(nn_edge.input)

    # DIFF
    img_manip = pipeline.create(dai.node.ImageManip)
    img_manip.initialConfig.setResize(720, 720) 
    img_manip.setMaxOutputFrameSize(720*720*3)

    preview.link(img_manip.inputImage)

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
    preview.link(nn_concat.inputs['img2'])
    manipRight.out.link(nn_concat.inputs['img3'])

   
    blur = pipeline.create(ReshapeNNOutputBlur).build(nn_out=nn_blur.out)
    blur = pipeline.create(Display).build(frame=blur.output)
    blur.setName("Blur")
    
    edge = pipeline.create(ReshapeNNOutputEdge).build(nn_out=nn_edge.out)
    edge = pipeline.create(Display).build(frame=edge.output)
    edge.setName("Edge")
    
    diff = pipeline.create(ColorizeDiff).build(nnOut=nn_diff.out)
    diff = pipeline.create(Display).build(frame=diff.output)
    diff.setName("Diff")
    
    concat = pipeline.create(ReshapeNNOutputConcat).build(nn_out=nn_concat.out)
    concat = pipeline.create(Display).build(frame=concat.output)
    concat.setName("Concat")    
    
    color = pipeline.create(Display).build(frame=preview)
    color.setName("Color")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")