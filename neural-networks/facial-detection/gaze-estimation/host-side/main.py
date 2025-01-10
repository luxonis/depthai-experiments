import argparse
from pathlib import Path
import blobconverter
import depthai as dai

from bbox_processing import BboxProcessing
from face_processing import FaceProcessing
from landmarks_processing import LandmarksProcessing
from gaze_estimation import GazeEstimation
from display import Display
from batching_neural_network import BatchingNeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="Prevent debug output"
)
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video file to be used for inference (conflicts with -cam)",
)
parser.add_argument("-laz", "--lazer", action="store_true", help="Lazer mode")
args = parser.parse_args()

debug = not args.no_debug
camera = not args.video

print("Creating pipeline...")
with dai.Pipeline() as pipeline:
    video_size = (300, 300)
    if camera:
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(video_size)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        fps = cam.getFps()
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(str(Path(args.video).resolve().absolute()))
        replay.setLoop(False)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setSize(video_size)
        fps = replay.getFps()
    video_output = cam.preview if camera else replay.out

    face_nn = pipeline.create(dai.node.NeuralNetwork)
    face_nn.setBlobPath(
        blobconverter.from_zoo(name="face-detection-retail-0004", shaves=4)
    )
    video_output.link(face_nn.input)

    bbox_processing = pipeline.create(BboxProcessing).build(video_output, face_nn.out)
    face_processing = pipeline.create(FaceProcessing).build(
        bbox_processing.output_img, bbox_processing.output_bboxes
    )

    land_nn = pipeline.create(BatchingNeuralNetwork).build()
    land_nn.setBlobPath(
        blobconverter.from_zoo(name="landmarks-regression-retail-0009", shaves=4)
    )
    land_nn.setInputShapeLen(2)
    face_processing.output_landmarks.link(land_nn.input)

    pose_nn = pipeline.create(BatchingNeuralNetwork).build()
    pose_nn.setBlobPath(
        blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=4)
    )
    pose_nn.setInputShapeLen(2)
    face_processing.output_pose.link(pose_nn.input)

    landmarks_processing = pipeline.create(LandmarksProcessing).build(
        bbox_processing.output_img,
        land_nn.out,
        face_processing.output_face,
        pose_nn.out,
    )

    gaze_nn = pipeline.create(BatchingNeuralNetwork).build()
    path = blobconverter.from_zoo(
        "gaze-estimation-adas-0002",
        shaves=4,
        compile_params=[
            "-iop head_pose_angles:FP16,right_eye_image:U8,left_eye_image:U8"
        ],
        version="2021.4",
    )
    gaze_nn.setBlobPath(path)
    gaze_nn.setInputShapeLen(2)
    landmarks_processing.output_gaze.link(gaze_nn.input)

    gaze_estimation = pipeline.create(GazeEstimation).build(
        bbox_processing.output_img,
        gaze_nn.out,
        landmarks_processing.output_right_bbox,
        landmarks_processing.output_left_bbox,
        bbox_processing.output_bboxes,
        landmarks_processing.output_nose,
        landmarks_processing.output_pose,
    )
    gaze_estimation.set_debug(debug)
    gaze_estimation.set_laser(args.lazer)
    gaze_estimation.set_camera(camera)

    display = pipeline.create(Display).build(gaze_estimation.output, camera)

    print("Starting pipeline...")
    pipeline.run()
    print("Pipeline ended.")
