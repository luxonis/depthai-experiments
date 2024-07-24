from pathlib import Path
import depthai as dai
from host_efficientDet import EfficientDet

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(320, 320)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFp16(True)  # Model requires FP16 input

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(Path("model/efficientdet_lite0_2021.3_6shaves.blob").resolve().absolute())
    nn.setNumInferenceThreads(2)
    cam.preview.link(nn.input)

    efficient_det = pipeline.create(EfficientDet).build(
        preview=cam.preview,
        nn=nn.out
    )
    efficient_det.inputs["preview"].setBlocking(False)
    efficient_det.inputs["preview"].setMaxSize(4)
    efficient_det.inputs["nn"].setBlocking(False)
    efficient_det.inputs["nn"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
