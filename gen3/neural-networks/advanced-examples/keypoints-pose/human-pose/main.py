import depthai as dai
import argparse

from host_human_pose import HumanPose
from pathlib import Path
from parsing.hrnet_parser import HRNetParser

device = dai.Device()
platform = device.getPlatform()
model_description = dai.NNModelDescription(modelSlug="lite-hrnet", platform=platform.name, modelVersionSlug="18-coco-256x192")
archive_path = dai.getModelFromZoo(model_description)

parser = argparse.ArgumentParser()
parser.add_argument('-vid', '--video', type=str
                    , help="Path to video file to be used for inference (otherwise uses the DepthAI color camera)")
args = parser.parse_args()

visualizer = dai.RemoteConnection()
with dai.Pipeline(device) as pipeline:
    output_type = dai.ImgFrame.Type.BGR888p if platform == dai.Platform.RVC2 else dai.ImgFrame.Type.BGR888i
    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setOutFrameType(output_type)
        replay.setSize(192*5, 256*5)
        video_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        video_out = cam.requestOutput((192*5, 256*5), output_type)
    
    manip = pipeline.create(dai.node.ImageManipV2)
    manip.initialConfig.setOutputSize(192, 256)
    video_out.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork).build(
        input=manip.out,
        nnArchive=dai.NNArchive(archive_path)
    )
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    parser = pipeline.create(HRNetParser)
    parser.setScoreThreshold(0.0) # Do not prune any keypoints
    nn.out.link(parser.input)

    human_pose = pipeline.create(HumanPose).build(parser.out)
    human_pose.inputs["keypoints"].setBlocking(False)
    human_pose.inputs["keypoints"].setMaxSize(2)

    visualizer.addTopic("Color Camera", video_out)
    visualizer.addTopic("Human Keypoints", human_pose.output_keypts)
    visualizer.addTopic("Human Pose", human_pose.output_pose)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
    print("Pipeline finished.")
