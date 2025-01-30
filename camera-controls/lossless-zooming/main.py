import depthai as dai

from depthai_nodes import YuNetParser
from host_crop_face import CropFace
from host_fps_drawer import FPSDrawer
from host_display import Display

full_shape = (3840, 2160)
hd_shape = (1920, 1080)
nn_shape = (320, 320)

model_description = dai.NNModelDescription(
    modelSlug="yunet", platform="RVC2", modelVersionSlug="320x320"
)
archive_path = dai.getModelFromZoo(model_description, useCached=True)

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam.initialControl.setManualFocus(130)
    cam_output = cam.requestOutput(full_shape, dai.ImgFrame.Type.YUV420p)

    nn = pipeline.create(dai.node.NeuralNetwork).build(
        input=cam.requestOutput(
            nn_shape, dai.ImgFrame.Type.BGR888p, dai.ImgResizeMode.STRETCH
        ),
        nnArchive=dai.NNArchive(archive_path),
    )
    nn.input.setBlocking(False)

    parser = pipeline.create(YuNetParser)
    nn.out.link(parser.input)

    crop_face = pipeline.create(CropFace).build(parser.out)

    crop_manip = pipeline.create(dai.node.ImageManip)
    crop_manip.initialConfig.setResize(*hd_shape)
    crop_manip.setMaxOutputFrameSize(hd_shape[0] * hd_shape[1] * 3)
    crop_face.output.link(crop_manip.inputConfig)
    cam_output.link(crop_manip.inputImage)
    crop_manip.inputImage.setBlocking(False)

    fps_drawer = pipeline.create(FPSDrawer).build(nn.passthrough)

    display_zoom = pipeline.create(Display).build(crop_manip.out)
    display_full = pipeline.create(Display).build(fps_drawer.output)
    display_zoom.setName("Zoom")
    display_full.setName("Full")

    print("Pipeline created.")
    pipeline.run()
