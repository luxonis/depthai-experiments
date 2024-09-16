import depthai as dai
from host_people_counter import PeopleCounter

model_description = dai.NNModelDescription(modelSlug="person-detection-retail", platform="RVC2")
archivePath = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archivePath)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 720)
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(544, 320)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    manip.out.link(nn.input)

    classification = pipeline.create(PeopleCounter).build(
        preview=cam.preview,
        nn=nn.out,
    )

    print("Pipeline created.")
    pipeline.run()
