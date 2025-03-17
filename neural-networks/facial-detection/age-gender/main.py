from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.host_sync import DetectionsAgeGenderSync
from utils.annotation_node import AnnotationNode
from depthai_nodes.node import CropConfigsCreatorNode

_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform()

FPS = 15
frame_type = dai.ImgFrame.Type.BGR888p
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 30

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    if args.media_path:
        replay_node = pipeline.create(dai.node.ReplayVideo)
        replay_node.setReplayVideoFile(Path(args.media_path))
        replay_node.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay_node.setLoop(True)

        video_resize_node = pipeline.create(dai.node.ImageManipV2)
        video_resize_node.initialConfig.setOutputSize(1280, 960)
        video_resize_node.initialConfig.setFrameType(frame_type)
        replay_node.out.link(video_resize_node.inputImage)

        input_node = video_resize_node.out
    else:
        camera_node = pipeline.create(dai.node.Camera).build()
        input_node = camera_node.requestOutput((1280, 960), frame_type, fps=FPS)

    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(640, 480)
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)
    input_node.link(resize_node.inputImage)

    face_detection_node: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(resize_node.out, "luxonis/yunet:640x480")

    config_sender_node = pipeline.create(CropConfigsCreatorNode).build(
        face_detection_node.out, 1280, 960, 62, 62, 10
    )

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(True)
    crop_node.inputConfig.setBlocking(True)
    crop_node.inputImage.setBlocking(True)
    crop_node.inputConfig.setMaxSize(10)
    crop_node.inputImage.setMaxSize(10)
    config_sender_node.config_output.link(crop_node.inputConfig)
    input_node.link(crop_node.inputImage)

    age_gender_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, "luxonis/age-gender-recognition:62x62"
    )

    sync_node = pipeline.create(DetectionsAgeGenderSync)
    input_node.link(sync_node.passthrough_input)
    config_sender_node.detections_output.link(sync_node.detections_input)
    age_gender_node.getOutput(0).link(sync_node.age_input)
    age_gender_node.getOutput(1).link(sync_node.gender_input)

    annotation_node = pipeline.create(AnnotationNode)
    sync_node.out.link(annotation_node.input)

    # crop_q = crop_node.out.createOutputQueue()
    # detections_q = config_sender_node.detections_output.createOutputQueue()
    # frame_q = input_node.createOutputQueue()

    visualizer.addTopic("Video", sync_node.out_frame)
    visualizer.addTopic("Text", annotation_node.output)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
        # print("--------------------")
        # detections_msg = detections_q.get()
        # frame_msg = frame_q.get()
        # frame = frame_msg.getCvFrame()
        # detections = detections_msg.detections

        # print("frame seq num", frame_msg.getSequenceNum())
        # print("detections seq num", detections_msg.getSequenceNum())
        # for i, detection in enumerate(detections):
        #     crop_msg = crop_q.get()
        #     crop_frame = crop_msg.getCvFrame()
        #     print("crop seq num", crop_msg.getSequenceNum())
        #     cv2.imshow(f"crop_{i}", crop_frame)

        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) == ord("q"):
        #     break
