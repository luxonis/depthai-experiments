import depthai as dai

from host_road_segmentation import RoadSegmentation
from host_fps_drawer import FPSDrawer
from host_display import Display

modelDescription = dai.NNModelDescription(modelSlug="road-segmentation-adas", platform="RVC2", modelVersionSlug="0001-896x512")
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)
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

    fps_drawer = pipeline.create(FPSDrawer).build(road_segmentation.output)

    display = pipeline.create(Display).build(fps_drawer.output)
    display.setName("Road Segmentation")

    print("Pipeline created.")
    pipeline.run()
