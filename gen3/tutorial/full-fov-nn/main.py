import depthai as dai
import blobconverter
from host_full_fov import FullFOV

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setInterleaved(False)
    cam.setIspScale(1, 5)  # 4056x3040 -> 812x608
    cam.setPreviewSize(812, 608)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    # Slightly lower FPS to avoid lag, as ISP takes more resources at 12MP
    cam.setFps(25)

    # Because of dai limitations only one manip can be linked in the neutral network
    #  however we can create 3 "fake" manips to display and make it seem like the pipeline is
    #  dynamically linking the correct manip in the NN
    nn_manip = pipeline.create(dai.node.ImageManip)
    nn_manip.setMaxOutputFrameSize(270000)  # 300x300x3
    nn_manip.initialConfig.setResize(300, 300)
    cam.preview.link(nn_manip.inputImage)

    crop_manip = pipeline.create(dai.node.ImageManip)
    crop_manip.setMaxOutputFrameSize(270000)
    crop_manip.initialConfig.setResize(300, 300)
    cam.preview.link(crop_manip.inputImage)

    letterbox_manip = pipeline.create(dai.node.ImageManip)
    letterbox_manip.setMaxOutputFrameSize(270000)
    letterbox_manip.initialConfig.setResizeThumbnail(300, 300)
    cam.preview.link(letterbox_manip.inputImage)

    stretch_manip = pipeline.create(dai.node.ImageManip)
    stretch_manip.setMaxOutputFrameSize(270000)
    stretch_manip.setKeepAspectRatio(False)
    stretch_manip.initialConfig.setResize(300, 300)
    cam.preview.link(stretch_manip.inputImage)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=5))
    nn_manip.out.link(nn.input)

    full_fov = pipeline.create(FullFOV).build(
        isp=cam.isp,
        crop_manip=crop_manip.out,
        letterbox_manip=letterbox_manip.out,
        stretch_manip=stretch_manip.out,
        nn_manip=nn_manip.out,
        nn=nn.out,
        manip_config=nn_manip.inputConfig.createInputQueue()
    )

    print("Pipeline created.")
    pipeline.run()
