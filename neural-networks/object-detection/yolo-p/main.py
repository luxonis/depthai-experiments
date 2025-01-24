from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()

model_reference = "luxonis/yolo-p:bdd100k-320x320"

if args.fps_limit and args.media_path:
    args.fps_limit = None
    print(
        "WARNING: FPS limit is set but media path is provided. FPS limit will be ignored."
    )

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription(model_reference)

    platform = device.getPlatform().name
    print(f"Platform: {platform}")

    model_description.platform = platform
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    frame_type = (
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        replay.setFps(10 if platform == "RVC2" else 20)

    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            (1280, 720), dai.ImgFrame.Type.NV12, fps=args.fps_limit
        )
    input_node = replay.out if args.media_path else cam_out

    imageManip = pipeline.create(dai.node.ImageManipV2)
    imageManip.setMaxOutputFrameSize(
        nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
    )
    imageManip.initialConfig.setOutputSize(
        nn_archive.getInputWidth(), nn_archive.getInputHeight()
    )
    imageManip.initialConfig.setFrameType(frame_type)
    input_node.link(imageManip.inputImage)

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        imageManip.out, nn_archive
    )

    annotation_node = pipeline.create(AnnotationNode)
    detection_nn.getOutput(0).link(annotation_node.input_detections)
    detection_nn.getOutput(1).link(annotation_node.input_road_segmentations)
    detection_nn.getOutput(2).link(annotation_node.input_lane_segmentations)
    input_node.link(annotation_node.input_frame)

    visualizer.addTopic(
        "Road Segmentation", annotation_node.out_segmentations, "images"
    )
    visualizer.addTopic("Detections", annotation_node.out_detections, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
