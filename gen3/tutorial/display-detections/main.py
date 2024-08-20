import depthai as dai
from host_display_detections import DisplayDetections

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
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

    nn = pipeline.create(dai.node.DetectionNetwork).build(input=crop_nn.out, nnArchive=nn_archive)

    display = pipeline.create(DisplayDetections).build(
        full=cam.preview,
        square=crop_square.out,
        passthrough=crop_nn.out,
        nn=nn.out
    )

    print("Pipeline created.")
    pipeline.run()
