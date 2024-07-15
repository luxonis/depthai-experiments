import depthai as dai
import blobconverter
from host_people_counter import PeopleCounter

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(1080, 720)
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(544, 320)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blobconverter.from_zoo(name="person-detection-retail-0013", zoo_type="intel", version="2021.4"))
    manip.out.link(nn.input)

    classification = pipeline.create(PeopleCounter).build(
        preview=cam.preview,
        nn=nn.out,
    )

    print("Pipeline created.")
    pipeline.run()
