import depthai as dai
from utils import PersonDetection, YUV2BGR
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ip", type=str, default=None, help="Device IP, if None use the one connected."
)
parser.add_argument(
    "--api",
    type=str,
    default="",
    help="API key of the HubAI team if model is private",
)
parser.add_argument(
    "--include_vehicles",
    action="store_true",
    help="Bool if should use model which also predicts vehicles",
)
parser.add_argument(
    "--video_path",
    type=str,
    default=None,
    help="If path to the video specified this will be used for inference instead of live feed.",
)
args = parser.parse_args()

nn_input_shape = (256, 192)
modelDescription = dai.NNModelDescription(
    modelSlug="thermalpersonvehicledetection",
    modelVersionSlug="personvehicle" if args.include_vehicles else "person",
    platform="RVC2",
    modelInstanceSlug=f"{nn_input_shape[1]}x{nn_input_shape[0]}",
)
archivePath = dai.getModelFromZoo(modelDescription, apiKey=args.api)
nn_archive = dai.NNArchive(archivePath)

device = dai.Device(dai.DeviceInfo(args.ip))
with dai.Pipeline(device) as pipeline:
    if args.video_path:
        nn_input_node = pipeline.create(dai.node.ReplayVideo)
        nn_input_node.setReplayVideoFile(Path(args.video_path))
        nn_input_node.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        nn_input_node.setSize(*nn_input_shape)
        nn_input_node.setLoop(False)
    else:
        cam = pipeline.create(dai.node.Thermal)

        nn_input_node = pipeline.create(YUV2BGR)

        # TODO: check why this isn't working instead of YUV2BGR
        # nn_input_node = pipeline.create(dai.node.ImageManip)
        # nn_input_node.initialConfig.setResize(nn_input_shape[0], nn_input_shape[1])
        # nn_input_node.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        cam.color.link(nn_input_node.input)

    # Play with setConfidenceThreshold() to get better results
    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setNNArchive(nn_archive)
    nn.setConfidenceThreshold(0.35)
    nn_input_node.out.link(nn.input)

    output = nn.passthrough

    pipeline.create(PersonDetection).build(
        img_frame=output,
        detections=nn.out,
    )

    print("Pipeline created")
    pipeline.run()
    print("Pipeline finished")
