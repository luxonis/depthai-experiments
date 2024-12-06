import depthai as dai

from depthai_nodes import SegmentationParser
from host_depth_segmentation import DepthSegmentation
from host_fps_drawer import FPSDrawer
from host_display import Display

model_description = dai.NNModelDescription(modelSlug="deeplab-v3-plus", platform="RVC2", modelVersionSlug="person-256x256")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

output_shape = (512, 512)
mono_shape = (640, 400)
nn_shape = (256, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_output = color.requestOutput(output_shape, dai.ImgFrame.Type.BGR888p)

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*nn_shape)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(nn_shape[0] * nn_shape[1] * 3)
    color_output.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork).build(
        nnArchive=nn_archive,
        input=manip.out
    )
    nn.input.setBlocking(False)
    nn.input.setMaxSize(1)

    parser = pipeline.create(SegmentationParser)
    nn.out.link(parser.input)
    parser.setBackgroundClass(True)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(mono_shape),
        right=right.requestOutput(mono_shape)
    )
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*output_shape)

    depth_segmentation = pipeline.create(DepthSegmentation).build(
        preview=color_output,
        mask=parser.out,
        disparity=stereo.disparity,
        max_disparity=stereo.initialConfig.getMaxDisparity()
    )

    fps_drawer = pipeline.create(FPSDrawer).build(depth_segmentation.output)
    display = pipeline.create(Display).build(fps_drawer.output)
    display.setName("Depth Segmentation")

    print("Pipeline created.")
    pipeline.run()
