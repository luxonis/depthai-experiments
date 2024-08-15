import depthai as dai
from host_qr_scanner import QRScanner

# if model changed change README
model_description = dai.NNModelDescription(modelSlug="qrdet", platform="RVC2")
archivePath = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(1080, 1080)
    cam.setInterleaved(False)
    cam.initialControl.setManualFocus(145)
    cam.setFps(20)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(512, 288) 
    manip.initialConfig.setKeepAspectRatio(False)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.DetectionNetwork).build(nnArchive=nn_archive, input=manip.out, confidenceThreshold=0.3)
    nn.input.setMaxSize(1)
    nn.input.setBlocking(False)

    scanner = pipeline.create(QRScanner).build(
        preview=cam.preview,
        nn=nn.out
    )
    scanner.inputs["detections"].setBlocking(False)
    scanner.inputs["detections"].setMaxSize(2)
    scanner.inputs["preview"].setBlocking(False)
    scanner.inputs["preview"].setMaxSize(2)

    print("Pipeline created.")
    pipeline.run()
