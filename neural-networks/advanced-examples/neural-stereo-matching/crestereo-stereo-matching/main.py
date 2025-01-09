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

parser.add_argument('-nn', '--nn-choice', type=str, choices=choices, default=choices[-1],
                    help=f"Crestereo model to be used for inference. By default the bigger model is chosen.")
args = parser.parse_args()

model = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription(args.nn_choice, platform.name)))
visualizer = dai.RemoteConnection()
with dai.Pipeline(device) as pipeline:
    FPS_CAP = 2 if platform == dai.Platform.RVC2 else 15
    print("Creating pipeline...")
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_steam = left.requestOutput((640,400), type=dai.ImgFrame.Type.NV12, fps=FPS_CAP)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_steam = right.requestOutput((640,400), type=dai.ImgFrame.Type.NV12, fps=FPS_CAP)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_steam,
        right=right_steam,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT
    )

    model_input_shape = model.getInputSize()
    manip_left = pipeline.create(dai.node.ImageManipV2)
    manip_left.initialConfig.setOutputSize(*model_input_shape)
    manip_left.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    stereo.rectifiedLeft.link(manip_left.inputImage)

    manip_right = pipeline.create(dai.node.ImageManipV2)
    manip_right.initialConfig.setOutputSize(*model_input_shape)
    manip_right.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    stereo.rectifiedRight.link(manip_right.inputImage)

    nn = pipeline.create(nodes.ParsingNeuralNetwork)
    nn.setNNArchive(model)
    manip_left.out.link(nn.inputs["left"])
    manip_right.out.link(nn.inputs["right"])

    disparity_coloring = pipeline.create(DepthColorTransform).build(stereo.disparity)
    disparity_coloring.setColormap(cv2.COLORMAP_PLASMA)

    visualizer.addTopic("Stereo Disparity", disparity_coloring.output)
    visualizer.addTopic("NN", nn.out)
    
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        if visualizer.waitKey(1) == "q":
            print("Q pressed. Stopping the pipeline.")
            pipeline.stop()
