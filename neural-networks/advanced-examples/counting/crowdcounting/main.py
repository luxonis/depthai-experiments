import argparse
from os.path import isfile
from pathlib import Path

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from download import download_vids
from host_node.host_depth_color_transform import DepthColorTransform
from host_node.overlay_frames import OverlayFrames
from host_node.visualize_detections_v2 import VisualizeDetectionsV2
from nn_configs import NN_CONFIGS
from visualize_crowd_count import VisualizeCrowdCount

device = dai.Device()


parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn",
    "--neural-network",
    type=str,
    choices=[
        "sha_small",
        "sha_medium",
        "sha_large",
        "sha_xlarge",
        "shb_small",
        "shb_medium",
        "shb_large",
        "shb_xlarge",
        "qnrf_small",
        "qnrf_medium",
        "qnrf_large",
        "qnrf_xlarge",
    ],
    default="sha_medium",
    help="Choose the neural network model used for crowd counting. Default: sha_medium",
)
parser.add_argument(
    "-fps",
    "--frames-per-second",
    type=float,
    help="Set the frames per second for the video. Default: 1",
    default=1,
)
parser.add_argument(
    "-v",
    "--video-path",
    type=str,
    help="Path to the video input for inference. Default: vids/virat.mp4",
    default="vids/vid4.mp4",
)
parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    help="Use the camera for inference instead of video. Default: False",
)
args = parser.parse_args()

# Download test videos
if (
    not isfile(Path("vids/virat.mp4").resolve().absolute())
    or not isfile(Path("vids/vid1.mp4").resolve().absolute())
    or not isfile(Path("vids/vid2.mp4").resolve().absolute())
    or not isfile(Path("vids/vid3.mp4").resolve().absolute())
    or not isfile(Path("vids/vid4.mp4").resolve().absolute())
):
    download_vids()
video_source = Path(args.video_path).resolve().absolute()


nn_config = NN_CONFIGS[args.neural_network]

NN_SIZE = nn_config["nn_size"]
VIDEO_SIZE = (1280, 720)
FPS = args.frames_per_second

model_description = dai.NNModelDescription(
    modelSlug=nn_config["model_slug"],
    platform=device.getPlatform().name,
    modelVersionSlug=nn_config["version_slug"],
)
archive_path = dai.getModelFromZoo(model_description, useCached=True)
nn_archive = dai.NNArchive(archive_path)

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.camera:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=FPS)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(video_source)
        replay.setSize(VIDEO_SIZE)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setFps(FPS)
        color_out = replay.out

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResizeThumbnail(NN_SIZE)
    manip.setMaxOutputFrameSize(NN_SIZE[0] * NN_SIZE[1] * 3)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.inputImage.setBlocking(True)
    color_out.link(manip.inputImage)

    nn = pipeline.create(ParsingNeuralNetwork).build(manip.out, nn_archive)

    visualize_crowd_count = pipeline.create(VisualizeCrowdCount).build(nn.out)

    visualize_detections = pipeline.create(VisualizeDetectionsV2).build(nn.out)

    color_transform = pipeline.create(DepthColorTransform).build(
        visualize_detections.output_mask
    )

    map_resize = pipeline.create(dai.node.ImageManipV2)
    map_resize.initialConfig.addResize(*VIDEO_SIZE)
    map_resize.setMaxOutputFrameSize(VIDEO_SIZE[0] * VIDEO_SIZE[1] * 3)
    color_transform.output.link(map_resize.inputImage)

    overlay_frames = pipeline.create(OverlayFrames).build(color_out, map_resize.out)

    visualizer.addTopic("Camera", color_out)
    visualizer.addTopic("Segmentation", overlay_frames.output)
    visualizer.addTopic("Predicted count", visualize_crowd_count.output)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
    print("Pipeline finished.")
