from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
from utils.annotation_node import AnnotationNode
from utils.arguments import initialize_argparser
from utils.host_process_detections import CropConfigsCreator

_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform()

FPS = 30
frame_type = dai.ImgFrame.Type.BGR888i
if "RVC2" in str(platform):
    frame_type = dai.ImgFrame.Type.RGB888p

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    if args.media_path:
        replay_node = pipeline.create(dai.node.ReplayVideo)
        replay_node.setReplayVideoFile(Path(args.media_path))
        replay_node.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay_node.setLoop(True)

        video_resize_node = pipeline.create(dai.node.ImageManipV2)
        video_resize_node.initialConfig.setOutputSize(640, 480)
        video_resize_node.initialConfig.setFrameType(frame_type)

        replay_node.out.link(video_resize_node.inputImage)

        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((640, 480), frame_type, fps=FPS)

    face_detection_node: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(input_node, "luxonis/yunet:640x480")
    face_detection_node.input.setBlocking(True)

    detection_process_node = pipeline.create(CropConfigsCreator).build(
        face_detection_node.out, (640, 480), (60, 60)
    )

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(True)
    crop_node.inputConfig.setMaxSize(30)
    crop_node.inputImage.setMaxSize(30)
    crop_node.setNumFramesPool(30)

    detection_process_node.config_output.link(crop_node.inputConfig)
    input_node.link(crop_node.inputImage)

    head_pose_node = pipeline.create(dai.node.NeuralNetwork)
    head_pose_node.setFromModelZoo(
        dai.NNModelDescription("luxonis/head-pose-estimation:60x60"), useCached=True
    )
    crop_node.out.link(head_pose_node.input)

    head_pose_node.input.setBlocking(True)

    sync_detection_node = pipeline.create(GatherData).build(FPS)
    detection_process_node.detections_output.link(sync_detection_node.input_reference)
    head_pose_node.out.link(sync_detection_node.input_data)

    frame_sync_node = pipeline.create(GatherData).build(FPS, lambda x: 1)
    input_node.link(frame_sync_node.input_reference)
    sync_detection_node.out.link(frame_sync_node.input_data)

    annotation_node = pipeline.create(AnnotationNode)
    frame_sync_node.out.link(annotation_node.input)

    visualizer.addTopic("annotation", annotation_node.output_annotation)
    visualizer.addTopic("frame", annotation_node.output_frame)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
