import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from typing import Tuple
from utils.host_triangulation import Triangulation
from utils.arguments import initialize_argparser


_, args = initialize_argparser()

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

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    device_platform = device.getPlatform()
    rvc2 = device_platform == dai.Platform.RVC2
    model_dimension = (320, 240) if rvc2 else (640, 480)
    # faceDet_modelDescription = dai.NNModelDescription(f"luxonis/yunet:{model_dimension[0]}x{model_dimension[1]}")
    faceDet_modelDescription = dai.NNModelDescription(
        modelSlug="yunet",
        platform=device.getPlatform().name,
        modelVersionSlug=f"{model_dimension[0]}x{model_dimension[1]}",
    )
    # faceDet_modelDescription.platform = device.getPlatformAsString()
    faceDet_nnarchive = dai.NNArchive(dai.getModelFromZoo(faceDet_modelDescription))

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

    visualizer.addTopic("Face Left", face_left, "left")
    visualizer.addTopic("Left Detections", triangulation.annot_left, "left")
    visualizer.addTopic("Face Right", face_right, "right")
    visualizer.addTopic("Right Detections", triangulation.annot_right, "right")
    visualizer.addTopic("Combined", triangulation.combined_frame, "combined")
    visualizer.addTopic("Keypoints", triangulation.combined_keypoints, "combined")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
