import depthai as dai
import depthai_nodes as nodes

from host_triangulation import Triangulation
from host_display import Display


device = dai.Device()
device_platform = device.getPlatform()

rvc2 = device_platform == dai.Platform.RVC2
model_dimension = (320, 240) if rvc2 else (640, 480)
faceDet_modelDescription = dai.NNModelDescription(
    modelSlug="yunet", 
    platform=device.getPlatform().name, 
    modelVersionSlug=f"new-{model_dimension[1]}x{model_dimension[0]}"
)
faceDet_archivePath = dai.getModelFromZoo(faceDet_modelDescription)
faceDet_nnarchive = dai.NNArchive(faceDet_archivePath)

# Creates and connects nodes, once for the left camera and once for the right camera
def populate_pipeline(p: dai.Pipeline, left: bool, resolution: tuple[int, int])\
        -> tuple[dai.Node.Output, dai.Node.Output, dai.Node.Output]:
    socket = dai.CameraBoardSocket.CAM_B if left else dai.CameraBoardSocket.CAM_C
    cam = p.create(dai.node.Camera).build(socket)
    fps = 25 if rvc2 else 3
    cam_output = cam.requestOutput(resolution, type=dai.ImgFrame.Type.NV12, fps=fps)

    face_nn = p.create(nodes.ParsingNeuralNetwork).build(cam, faceDet_nnarchive, fps)

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
        resolution_number=model_dimension
    )

    display = pipeline.create(Display).build(triangulation.output)
    display.setName("Combined frame")

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")