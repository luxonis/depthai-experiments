# coding=utf-8
import os
import argparse
from pathlib import Path
import blobconverter
import depthai as dai
from face_recognition import FaceRecognition
from face_recognition_node import FaceRecognitionNode
from detections_recognitions_sync import DetectionsRecognitionsSync
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

VIDEO_SIZE = (1280, 800)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

face_det_model_description = dai.NNModelDescription(modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640")
face_det_archive_path = dai.getModelFromZoo(face_det_model_description)
face_det_nn_archive = dai.NNArchive(face_det_archive_path)

headpose_model_description = dai.NNModelDescription(modelSlug="head-pose-estimation-adas", platform="RVC2", modelVersionSlug="0001")
headpose_archive_path = dai.getModelFromZoo(headpose_model_description)
headpose_nn_archive = dai.NNArchive(headpose_archive_path)

face_rec_model_description = dai.NNModelDescription(modelSlug="face-recognition-arcface", platform="RVC2", modelVersionSlug="112x112")
face_rec_archive_path = dai.getModelFromZoo(face_rec_model_description)
face_rec_nn_archive = dai.NNArchive(face_rec_archive_path)

print("Creating pipeline...")
with dai.Pipeline() as pipeline:
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    # For ImageManip rotate you need input frame of multiple of 16
    cam.setPreviewSize(1280, 800)
    cam.setVideoSize(VIDEO_SIZE)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    # ImageManip as a workaround to have more frames in the pool.
    # cam.preview can only have 4 frames in the pool before it will
    # wait (freeze). Copying frames and setting ImageManip pool size to
    # higher number will fix this issue.
    copy_manip = pipeline.create(dai.node.ImageManip)
    copy_manip.setNumFramesPool(20)
    copy_manip.setMaxOutputFrameSize(1072*1072*3)
    cam.preview.link(copy_manip.inputImage)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    copy_manip.out.link(face_det_manip.inputImage)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    # face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_nn.setNNArchive(face_det_nn_archive)
    # Link Face ImageManip -> Face detection NN node
    face_det_manip.out.link(face_det_nn.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)

    face_det_nn.out.link(script.inputs['face_det_in'])
    # We also interested in sequence number for syncing
    face_det_nn.passthrough.link(script.inputs['face_pass'])

    copy_manip.out.link(script.inputs['preview'])

    with open(Path(__file__).parent / Path("script.py"), "r") as f:
        script.setScript(f.read())

    print("Creating Head pose estimation NN")

    headpose_manip = pipeline.create(dai.node.ImageManip)
    headpose_manip.initialConfig.setResize(60, 60)
    headpose_manip.inputConfig.setWaitForMessage(True)
    script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
    script.outputs['manip_img'].link(headpose_manip.inputImage)

    headpose_nn = pipeline.create(dai.node.NeuralNetwork)
    # headpose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
    headpose_nn.setNNArchive(headpose_nn_archive)
    headpose_manip.out.link(headpose_nn.input)

    headpose_nn.out.link(script.inputs['headpose_in'])
    headpose_nn.passthrough.link(script.inputs['headpose_pass'])

    print("Creating face recognition ImageManip/NN")

    face_rec_manip = pipeline.create(dai.node.ImageManip)
    face_rec_manip.initialConfig.setResize(112, 112)
    face_rec_manip.inputConfig.setWaitForMessage(True)

    script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
    script.outputs['manip2_img'].link(face_rec_manip.inputImage)

    face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
    # face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
    face_rec_nn.setNNArchive(face_rec_nn_archive)
    face_rec_manip.out.link(face_rec_nn.input)

    sync_node = pipeline.create(DetectionsRecognitionsSync).build()
    sync_node.set_camera_fps(cam.getFps())

    face_det_nn.out.link(sync_node.input_detections)
    face_rec_nn.out.link(sync_node.input_recognitions)

    facerec = FaceRecognition(databases, args.name)

    pipeline.create(FaceRecognitionNode).build(
        img_output=cam.video,
        detected_recognitions=sync_node.output,
        face_recognition=facerec
    )

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline ended.")
