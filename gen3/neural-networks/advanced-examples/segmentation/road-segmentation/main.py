import argparse
from os.path import isfile
from pathlib import Path

import cv2
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from download import download_vids
from host_node.host_depth_color_transform import DepthColorTransform
from host_node.overlay_frames import OverlayFrames

parser = argparse.ArgumentParser()
parser.add_argument(
    "-vid",
    "--video",
    default="vids/vid3.mp4",
    type=str,
    help="Path to video to use for inference. Default: vids/vid3.mp4",
)
parser.add_argument(
    "-cam",
    "--camera",
    default=False,
    action="store_true",
    help="Use camera for inference. Default: False",
)
args = parser.parse_args()


# Download test videos
if (
    not isfile(Path("vids/vid1.mp4").resolve().absolute())
    or not isfile(Path("vids/vid2.mp4").resolve().absolute())
    or not isfile(Path("vids/vid3.mp4").resolve().absolute())
):
    download_vids()

device = dai.Device()

model_description = dai.NNModelDescription(
    modelSlug="pp-liteseg",
    platform=device.getPlatform().name,
    modelVersionSlug="512x1024",
)
archive_path = dai.getModelFromZoo(model_description, useCached=True)
nn_archive = dai.NNArchive(archive_path)

NN_SIZE = (1024, 512)
FPS = 5 if device.getPlatform() == dai.Platform.RVC2 else 30

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.camera:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput(NN_SIZE, dai.ImgFrame.Type.BGR888p, fps=FPS)
    else:
        cam = pipeline.create(dai.node.ReplayVideo)
        cam.setReplayVideoFile(Path(args.video).resolve().absolute())
        color_out = cam.out

    nn_resize = pipeline.create(dai.node.ImageManipV2)
    nn_resize.initialConfig.addResize(*NN_SIZE)
    nn_resize.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    nn_resize.setMaxOutputFrameSize(NN_SIZE[0] * NN_SIZE[1] * 3)
    color_out.link(nn_resize.inputImage)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=nn_resize.out, nn_source=nn_archive
    )

    color_transform = pipeline.create(DepthColorTransform).build(nn.out)
    color_transform.setMaxDisparity(19)
    color_transform.setColormap(cv2.COLORMAP_JET)

    overlay_frames = pipeline.create(OverlayFrames).build(
        nn_resize.out, color_transform.output
    )

    visualizer.addTopic("Camera", nn_resize.out)
    visualizer.addTopic("Segmentation", overlay_frames.output)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
    print("Pipeline finished.")
