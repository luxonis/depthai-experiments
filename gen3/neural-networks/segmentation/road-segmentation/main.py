import depthai as dai
from host_road_segmentation import RoadSegmentation

modelDescription = dai.NNModelDescription(modelSlug="road-segmentation-adas", platform="RVC2", modelVersionSlug="0001-896x512") # private model
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)

nn_shape = (896, 512)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(nn_shape)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    cam.preview.link(nn.input)

    road_segmentation = pipeline.create(RoadSegmentation).build(
        preview=cam.preview,
        nn=nn.out
    )
    road_segmentation.inputs["preview"].setBlocking(False)
    road_segmentation.inputs["preview"].setMaxSize(4)

    print("Pipeline created.")
    pipeline.run()
