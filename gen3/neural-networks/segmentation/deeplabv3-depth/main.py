import depthai as dai

from host_depth_segmentation import DepthSegmentation
from host_fps_drawer import FPSDrawer
from host_display import Display

model_description = dai.NNModelDescription(modelSlug="deeplabv3", platform="RVC2", modelVersionSlug="256x256")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

nn_shape = (256, 256)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    color.requestOutput((256, 256), dai.ImgFrame.Type.BGR888p).link(nn.input)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput((400, 400), dai.ImgFrame.Type.BGR888p),
        right=right.requestOutput((400, 400), dai.ImgFrame.Type.BGR888p)
    )
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(400, 400)

    depth_segmentation = pipeline.create(DepthSegmentation).build(
        preview=color.requestOutput((400, 400), dai.ImgFrame.Type.BGR888p),
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
