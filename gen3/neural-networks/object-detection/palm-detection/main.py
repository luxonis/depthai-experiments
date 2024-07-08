import depthai as dai
from host_palm_detection import PalmDetection
import blobconverter

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(40)
    cam.setIspScale(2, 3) # 720P
    cam.setVideoSize(720, 720)
    cam.setPreviewSize(128, 128)
    cam.setInterleaved(False)

    model_nn = pipeline.create(dai.node.NeuralNetwork)
    model_nn.setBlobPath(blobconverter.from_zoo(name="palm_detection_128x128", zoo_type="depthai", shaves=6))
    model_nn.input.setBlocking(False)
    cam.preview.link(model_nn.input)

    palm_detection = pipeline.create(PalmDetection).build(
        preview=cam.video,
        palm_detections=model_nn.out
    )

    print("Pipeline created.")
    pipeline.run()
