import depthai as dai

from depthai_nodes import YuNetParser
from host_crop_face import CropFace
from host_fps_drawer import FPSDrawer
from host_display import Display

model_description = dai.NNModelDescription(modelSlug="yunet", platform="RVC2", modelVersionSlug="320x320")
archive_path = dai.getModelFromZoo(model_description, useCached=True)

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam.initialControl.setManualFocus(130)

    nn = pipeline.create(dai.node.NeuralNetwork).build(
        cam.requestOutput((320, 320), dai.ImgFrame.Type.BGR888p, dai.ImgResizeMode.STRETCH),
        dai.NNArchive(archive_path)
    )
    nn.input.setBlocking(False)

    parser = pipeline.create(YuNetParser)
    nn.out.link(parser.input)

    crop_face = pipeline.create(CropFace).build(parser.out)

    crop_manip = pipeline.create(dai.node.ImageManip)
    crop_manip.initialConfig.setResize(1920, 1080)
    crop_manip.setMaxOutputFrameSize(3110400)
    crop_face.output.link(crop_manip.inputConfig)
    cam.requestOutput((3840, 2160), dai.ImgFrame.Type.YUV420p).link(crop_manip.inputImage)
    crop_manip.inputImage.setBlocking(False)

    fps_drawer = pipeline.create(FPSDrawer).build(nn.passthrough)

    display_zoom = pipeline.create(Display).build(crop_manip.out)
    display_full = pipeline.create(Display).build(fps_drawer.output)
    display_zoom.setName("Zoom")
    display_full.setName("Full")

    print("Pipeline created.")
    pipeline.run()
