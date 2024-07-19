import depthai as dai
import blobconverter
from host_qr_scanner import QRScanner

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(1080, 1080)
    cam.setInterleaved(False)
    cam.initialControl.setManualFocus(145)
    # 30 fps because the cv2 QR decoder bottlenecks the pipeline
    cam.setFps(30)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(384, 384)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    nn.setConfidenceThreshold(0.3)
    nn.setBlobPath(blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai", shaves=6))
    nn.input.setMaxSize(1)
    nn.input.setBlocking(False)
    manip.out.link(nn.input)

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
