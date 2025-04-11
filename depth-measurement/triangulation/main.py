import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from typing import Tuple
from utils.host_triangulation import Triangulation
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

device_platform = device.getPlatform()
rvc2 = device_platform == dai.Platform.RVC2
model_dimension = (320, 240) if rvc2 else (640, 400)
faceDet_modelDescription = dai.NNModelDescription(
    modelSlug="yunet",
    platform=device.getPlatform().name,
    modelVersionSlug=f"{model_dimension[0]}x{model_dimension[1]}",
)
faceDet_archivePath = dai.getModelFromZoo(faceDet_modelDescription)
faceDet_nnarchive = dai.NNArchive(faceDet_archivePath)


# Creates and connects nodes, once for the left camera and once for the right camera
def populate_pipeline(
    p: dai.Pipeline, left: bool, resolution: Tuple[int, int]
) -> Tuple[dai.Node.Output, dai.Node.Output]:
    socket = dai.CameraBoardSocket.CAM_B if left else dai.CameraBoardSocket.CAM_C
    cam = p.create(dai.node.Camera).build(socket)
    fps = 25 if rvc2 else 3
    cam_output = cam.requestOutput(resolution, type=dai.ImgFrame.Type.NV12, fps=fps)

    face_nn = p.create(ParsingNeuralNetwork).build(cam, faceDet_nnarchive, fps)

    return cam_output, face_nn.out


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    face_left, face_nn_left = populate_pipeline(pipeline, True, model_dimension)
    face_right, face_nn_right = populate_pipeline(pipeline, False, model_dimension)

    triangulation = pipeline.create(Triangulation).build(
        face_left=face_left,
        face_right=face_right,
        face_nn_left=face_nn_left,
        face_nn_right=face_nn_right,
        device=device,
        resolution_number=model_dimension,
    )

    visualizer.addTopic("Face left", face_left)
    visualizer.addTopic("Face right", face_right)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
