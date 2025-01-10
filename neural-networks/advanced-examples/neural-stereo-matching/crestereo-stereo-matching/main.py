import argparse
import cv2
import depthai as dai
import depthai_nodes as nodes
from host_node.host_depth_color_transform import DepthColorTransform


device = dai.Device()
platform = device.getPlatform()
parser = argparse.ArgumentParser()

if platform == dai.Platform.RVC2:
    choices = ["luxonis/crestereo:iter2-120x160", "luxonis/crestereo:iter2-240x320"]
elif platform == dai.Platform.RVC4:
    choices = ["luxonis/crestereo:iter5-240x320", "luxonis/crestereo:iter4-360x640"]

parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=choices,
        default=choices[-1],
        help=f"Crestereo model to be used for inference. By default the bigger model is chosen.")
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

model = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription(args.nn_choice, platform.name)))
visualizer = dai.RemoteConnection()
with dai.Pipeline(device) as pipeline:
    fps_cap = args.fps_limit
    OUTPUT_TYPE = dai.ImgFrame.Type.BGR888p if platform == dai.Platform.RVC2 else dai.ImgFrame.Type.BGR888i
    print("Creating pipeline...")

    model_input_shape = model.getInputSize()
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(model_input_shape, type=dai.ImgFrame.Type.NV12, fps=fps_cap),
        right=right.requestOutput(model_input_shape, type=dai.ImgFrame.Type.NV12, fps=fps_cap),
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT
    )

    lr_sync = pipeline.create(dai.node.Sync)
    left.requestOutput(model_input_shape, type=OUTPUT_TYPE, fps=fps_cap).link(lr_sync.inputs["left"])
    right.requestOutput(model_input_shape, type=OUTPUT_TYPE, fps=fps_cap).link(lr_sync.inputs["right"])
    
    demux = pipeline.create(dai.node.MessageDemux)
    lr_sync.out.link(demux.input)

    nn = pipeline.create(nodes.ParsingNeuralNetwork)
    nn.setNNArchive(model)
    if platform == dai.Platform.RVC4:
        nn.setBackend("snpe")
        nn.setBackendProperties(
            {
                "runtime": "cpu",  # using "cpu" since the model is not quantized, use "dsp" if the model is quantized
                "performance_profile": "default",
            }
        )

    demux.outputs["left"].link(nn.inputs["left"])
    demux.outputs["right"].link(nn.inputs["right"])

    disparity_coloring = pipeline.create(DepthColorTransform).build(stereo.disparity)
    disparity_coloring.setColormap(cv2.COLORMAP_PLASMA)

    visualizer.addTopic("Stereo Disparity", disparity_coloring.output)
    visualizer.addTopic("NN", nn.out)
    
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        if visualizer.waitKey(1) == ord("q"):
            print("Q pressed. Stopping the pipeline.")
            pipeline.stop()
