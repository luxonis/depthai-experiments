# coding=utf-8
from pathlib import Path
import argparse

import depthai as dai
from fire_detection import FireDetection
from display import Display
from fps_counter import FPSCounter


parser = argparse.ArgumentParser()
parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="prevent debug output"
)

parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="The path of the video file used for inference (otherwise uses DepthAI 4K RGB camera)",
)

args = parser.parse_args()

debug = not args.no_debug

with dai.Pipeline() as pipeline:
    cam_size = (224, 224)
    if not args.video:
        cam = pipeline.create(dai.node.ColorCamera).build()
        cam.setPreviewSize(cam_size)
        cam.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_4_K
        )
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        img_output = cam.preview
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setLoop(False)
        replay.setSize(cam_size)
        replay.setReplayVideoFile(str(Path(args.video).resolve().absolute()))
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        img_output = replay.out

    neural_network = pipeline.create(dai.node.NeuralNetwork)
    neural_network.setBlobPath(str((Path(__file__).parent / "models/fire-detection_openvino_2021.2_5shave.blob").resolve().absolute()))
    neural_network.input.setBlocking(False)
    img_output.link(neural_network.input)

    nn_fps_counter = pipeline.create(FPSCounter).build(neural_network.out)
    nn_fps_counter.set_name("NN")

    camera_fps_counter = pipeline.create(FPSCounter).build(img_output)
    camera_fps_counter.set_name("Camera")

    fire_detection = pipeline.create(FireDetection).build(img_output, neural_network.out)

    if debug:
        display = pipeline.create(Display).build(fire_detection.output)
        display.set_window_name("Fire Detection")

    pipeline.run()
