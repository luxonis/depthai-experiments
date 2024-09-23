import depthai as dai
import depthai_nodes as nodes
from host_facemesh import Facemesh
from depthai_nodes import KeypointParser
from crop_detections import CropDetections
from host_display import Display


#TODO: fix for RVC4
device = dai.Device()
platform = device.getPlatform()
rvc2 = platform == dai.Platform.RVC2

yunet_input_size = (320,320) if rvc2 else (640,640)
yunet_model = dai.getModelFromZoo(dai.NNModelDescription(
    modelSlug="yunet",
    modelVersionSlug=f"{yunet_input_size[0]}x{yunet_input_size[1]}",
    platform=platform.name)
)
yunet_archive = dai.NNArchive(yunet_model)

face_landmark_model = dai.getModelFromZoo(dai.NNModelDescription(
    modelSlug="mediapipe-face-landmarker",
    modelVersionSlug="192x192",
    platform=platform.name)
)
face_landmark_archive = dai.NNArchive(face_landmark_model)

with dai.Pipeline(device) as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = cam.requestOutput(yunet_input_size, dai.ImgFrame.Type.BGR888p if rvc2 else dai.ImgFrame.Type.BGR888i, fps=15)

    face_nn = pipeline.create(dai.node.NeuralNetwork).build(output, yunet_archive)
    face_nn.input.setBlocking(False)
    face_nn.input.setMaxSize(2)

    yunet_parser = pipeline.create(nodes.YuNetParser)
    yunet_parser.setConfidenceThreshold(0.5)
    face_nn.out.link(yunet_parser.input)

    crop_detections = pipeline.create(CropDetections).build(face_nn=yunet_parser.out, manipv2=not rvc2)

    if rvc2:
        crop_face = pipeline.create(dai.node.ImageManip)
        crop_face.setMaxOutputFrameSize(3110400)
        crop_face.inputConfig.setWaitForMessage(False)
        crop_face.initialConfig.setResize(192, 192)
        crop_detections.out.link(crop_face.inputConfig)
        output.link(crop_face.inputImage)
    else:
        crop_face = pipeline.create(dai.node.ImageManipV2)
        crop_face.initialConfig.setOutputSize(192, 192)
        crop_face.initialConfig.setReusePreviousImage(False)
        crop_detections.out.link(crop_face.inputConfig)
        output.link(crop_face.inputImage)

    landmarks_nn = pipeline.create(dai.node.NeuralNetwork).build(crop_face.out, face_landmark_archive)
    landmarks_nn.setNumPoolFrames(4)
    landmarks_nn.input.setBlocking(False)
    landmarks_nn.input.setMaxSize(2)
    landmarks_nn.setNumInferenceThreads(2)

    landmarks_parser = pipeline.create(KeypointParser)
    landmarks_parser.setNumKeypoints(468)
    landmarks_parser.setScaleFactor(192)
    landmarks_nn.out.link(landmarks_parser.input)
    landmarks_parser.input.setBlocking(False)
    landmarks_parser.input.setMaxSize(2)

    facemesh = pipeline.create(Facemesh).build(
        preview=output,
        face_nn=yunet_parser.out,
        landmarks_nn=landmarks_parser.out
    )

    mesh_display = pipeline.create(Display).build(facemesh.output_mask)
    mesh_display.setName("Mesh")

    landmarks_display = pipeline.create(Display).build(facemesh.output_landmarks)
    landmarks_display.setName("Landmarks")

    print("Pipeline created.")
    pipeline.run()
