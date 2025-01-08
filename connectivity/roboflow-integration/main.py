import argparse
import blobconverter
import depthai as dai
from host_roboflow import Roboflow
from uploader import RoboflowUploader

parser = argparse.ArgumentParser()
parser.add_argument("-key", "--api-key", help="private API key copied from app.roboflow.com", required=True)
parser.add_argument("--workspace", help="name of the workspace in app.roboflow.com", required=True)
parser.add_argument("--dataset", help="name of the project in app.roboflow.com", required=True)
parser.add_argument(
    "-ai", "--autoupload-interval",
    help="automatically upload annotations every [SECONDS] seconds (when used with --autoupload-threshold)",
    default=0,
    type=float
)
parser.add_argument(
    "-at", "--autoupload-threshold",
    help="automatically upload annotations with confidence above [THRESHOLD] (when used with --autoupload-interval)",
    default=0,
    type=float
)
parser.add_argument(
    "-res", "--upload-res",
    help="upload annotated images in [WIDTHxHEIGHT] resolution, which can be useful to create dataset with high-resolution images",
    default="300x300",
    type=str,
)
args = parser.parse_args()

target_res = [int(x) for x in args.upload_res.split("x")]

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(target_res[0], target_res[1])
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setInterleaved(False)
    cam.setFps(30)
    cam.setPreviewKeepAspectRatio(False)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setKeepAspectRatio(False)
    manip.initialConfig.setResize(300, 300)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=5))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    manip.out.link(nn.input)

    uploader = RoboflowUploader(
        api_key=args.api_key,
        workspace_name=args.workspace,
        dataset_name=args.dataset
    )

    roboflow = pipeline.create(Roboflow).build(
        preview=cam.preview,
        nn=nn.out,
        uploader=uploader,
        target_resolution=target_res,
        auto_interval=args.autoupload_interval,
        auto_threshold=args.autoupload_threshold
    )
    roboflow.inputs["preview"].setBlocking(False)
    roboflow.inputs["preview"].setMaxSize(5)

    print("Pipeline created.")
    pipeline.run()
