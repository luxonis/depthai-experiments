import argparse
import cv2
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.host_depth_color_transform import DepthColorTransform

device = dai.Device()
platform = device.getPlatform()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

if platform == dai.Platform.RVC2:
    choices = ["luxonis/crestereo:iter2-160x120", "luxonis/crestereo:iter2-320x240"]
elif platform == dai.Platform.RVC4:
    choices = ["luxonis/crestereo:iter5-320x240", "luxonis/crestereo:iter4-640x360"]

parser.add_argument(
    "-m",
    "--model",
    type=str,
    choices=choices,
    default=choices[-1],
    help="Crestereo model to be used for inference. By default the bigger model is chosen.",
)
default_fps = 2 if platform == dai.Platform.RVC2 else 5
parser.add_argument(
    "-fps",
    "--fps_limit",
    help=f"FPS limit of the video. Default for the device is {default_fps}",
    required=False,
    default=default_fps,
    type=int,
)
args = parser.parse_args()

model = dai.NNArchive(
    dai.getModelFromZoo(dai.NNModelDescription(args.model, platform.name))
)
visualizer = dai.RemoteConnection()
with dai.Pipeline(device) as pipeline:
    fps_cap = args.fps_limit
    OUTPUT_TYPE = (
        dai.ImgFrame.Type.BGR888p
        if platform == dai.Platform.RVC2
        else dai.ImgFrame.Type.BGR888i
    )
    print("Creating pipeline...")

    model_input_shape = model.getInputSize()
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(
            model_input_shape, type=dai.ImgFrame.Type.NV12, fps=fps_cap
        ),
        right=right.requestOutput(
            model_input_shape, type=dai.ImgFrame.Type.NV12, fps=fps_cap
        ),
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )

    lr_sync = pipeline.create(dai.node.Sync)
    left.requestOutput(model_input_shape, type=OUTPUT_TYPE, fps=fps_cap).link(
        lr_sync.inputs["left"]
    )
    right.requestOutput(model_input_shape, type=OUTPUT_TYPE, fps=fps_cap).link(
        lr_sync.inputs["right"]
    )

    demux = pipeline.create(dai.node.MessageDemux)
    lr_sync.out.link(demux.input)

    nn = pipeline.create(ParsingNeuralNetwork)
    if platform == dai.Platform.RVC4:
        nn.setNNArchive(model)
        nn.setBackend("snpe")
        nn.setBackendProperties(
            {
                "runtime": "cpu",  # using "cpu" since the model is not quantized, use "dsp" if the model is quantized
                "performance_profile": "default",
            }
        )
    elif platform == dai.Platform.RVC2:
        nn.setNNArchive(model, numShaves=7)

    demux.outputs["left"].link(nn.inputs["left"])
    demux.outputs["right"].link(nn.inputs["right"])

    disparity_coloring = pipeline.create(DepthColorTransform).build(stereo.disparity)
    disparity_coloring.setColormap(cv2.COLORMAP_PLASMA)

    visualizer.addTopic("Stereo Disparity", disparity_coloring.out)
    visualizer.addTopic("NN", nn.out)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        if visualizer.waitKey(1) == ord("q"):
            print("Q pressed. Stopping the pipeline.")
            pipeline.stop()
