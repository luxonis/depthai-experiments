import depthai as dai

from host_depth_segmentation import DepthSegmentation
from host_fps_drawer import FPSDrawer
from host_display import Display

model_description = dai.NNModelDescription(modelSlug="deeplabv3", platform="RVC2", modelVersionSlug="256x256")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

output_shape = (400, 400)
nn_shape = (256, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    color_output = color.requestOutput(output_shape)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*nn_shape)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(nn_shape[0] * nn_shape[1] * 3)
    color_output.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    manip.out.link(nn.input)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(output_shape),
        right=right.requestOutput(output_shape)
    )
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*output_shape)

    depth_segmentation = pipeline.create(DepthSegmentation).build(
        preview=color_output,
        nn=nn.out,
        disparity=stereo.disparity,
        nn_shape=nn_shape,
        max_disparity=stereo.initialConfig.getMaxDisparity()
    )

    fps_drawer = pipeline.create(FPSDrawer).build(depth_segmentation.output)

    display = pipeline.create(Display).build(fps_drawer.output)
    display.setName("Depth Segmentation")

    print("Pipeline created.")
    pipeline.run()
