import depthai as dai
import blobconverter
from host_display_detections import DisplayDetections

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setInterleaved(False)
    cam.setIspScale(1, 3)  # 4k -> 720P
    cam.setPreviewSize(1280, 720)

    # Crop video to match aspect ratio of the detection network (1:1)
    crop_square = pipeline.create(dai.node.ImageManip)
    crop_square.initialConfig.setResize(720, 720)
    crop_square.setMaxOutputFrameSize(1555200) # 720 x 720 x 3
    cam.preview.link(crop_square.inputImage)

    # Crop video to match detection network
    crop_nn = pipeline.create(dai.node.ImageManip)
    crop_nn.initialConfig.setResize(300, 300)
    cam.preview.link(crop_nn.inputImage)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=5))
    crop_nn.out.link(nn.input)

    display = pipeline.create(DisplayDetections).build(
        full=cam.preview,
        square=crop_square.out,
        passthrough=crop_nn.out,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
