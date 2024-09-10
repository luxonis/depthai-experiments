import depthai as dai
import re
import depthai_nodes as nodes

from host_triangulation import Triangulation
from host_display import Display

device = dai.Device()
device_platform = device.getPlatform()

rvc2 = device_platform == dai.Platform.RVC2
model_dimension = 320 if rvc2 else 640
faceDet_modelDescription = dai.NNModelDescription(modelSlug="yunet", platform=device.getPlatform().name, modelVersionSlug=f"{model_dimension}x{model_dimension}")
faceDet_archivePath = dai.getModelFromZoo(faceDet_modelDescription)
faceDet_nnarchive = dai.NNArchive(faceDet_archivePath)

# Creates and connects nodes, once for the left camera and once for the right camera
def populate_pipeline(p: dai.Pipeline, left: bool, resolution: dai.MonoCameraProperties.SensorResolution)\
        -> tuple[dai.Node.Output, dai.Node.Output, dai.Node.Output]:
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.CAM_B if left else dai.CameraBoardSocket.CAM_C
    cam.setBoardSocket(socket)
    cam.setResolution(resolution)
    if rvc2:
        cam.setFps(25)

    face_manip = p.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(model_dimension, model_dimension)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    face_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    face_manip.setMaxOutputFrameSize(model_dimension*model_dimension*3)
    cam.out.link(face_manip.inputImage)

    face_nn = p.create(dai.node.NeuralNetwork).build(face_manip.out, faceDet_nnarchive)
    
    yunet_parser = p.create(nodes.YuNetParser)
    yunet_parser.setConfidenceThreshold(0.2)
    face_nn.out.link(yunet_parser.input)

    return face_manip.out, yunet_parser.out


with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

    face_left, face_nn_left = populate_pipeline(pipeline, True, resolution)
    face_right, face_nn_right = populate_pipeline(pipeline, False, resolution)

    triangulation = pipeline.create(Triangulation).build(
        face_left=face_left,
        face_right=face_right,
        face_nn_left=face_nn_left,
        face_nn_right=face_nn_right,
        device=device,
        resolution_number=int(re.findall(r"\d+", str(resolution))[0])
    )

    display = pipeline.create(Display).build(triangulation.output)
    display.setName("Combined frame")

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")