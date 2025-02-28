from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.process_keypoints import ProcessKeypointDetections
import cv2
import numpy as np

_, args = initialize_argparser()
# visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device) if args.device else dai.DeviceInfo())
platform = device.getPlatform()

FPS = 20
frame_type = dai.ImgFrame.Type.BGR888p
if "RVC4" in str(platform):
    frame_type = dai.ImgFrame.Type.BGR888i
    FPS = 30
else:
    raise RuntimeError("This demo is currently only supported on RVC4")

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

    detection_process_node = pipeline.create(ProcessKeypointDetections)
    detection_process_node.set_source_size(1280, 960)
    detection_process_node.set_target_size(60, 60)
    face_detection_node.out.link(detection_process_node.detections_input)

    left_config_sender_node = pipeline.create(dai.node.Script)
    left_config_sender_node.setScriptPath(
        Path(__file__).parent / "utils/config_sender_script.py"
    )
    left_config_sender_node.inputs["frame_input"].setBlocking(True)
    left_config_sender_node.inputs["config_input"].setBlocking(True)
    left_config_sender_node.inputs["frame_input"].setMaxSize(30)
    left_config_sender_node.inputs["config_input"].setMaxSize(30)

    input_node.link(left_config_sender_node.inputs["frame_input"])
    detection_process_node.left_config_output.link(
        left_config_sender_node.inputs["config_input"]
    )

    left_crop_node = pipeline.create(dai.node.ImageManipV2)
    left_crop_node.initialConfig.setReusePreviousImage(False)
    left_crop_node.inputConfig.setReusePreviousMessage(False)
    left_crop_node.inputImage.setReusePreviousMessage(False)

    left_config_sender_node.outputs["output_config"].link(left_crop_node.inputConfig)
    left_config_sender_node.outputs["output_frame"].link(left_crop_node.inputImage)

    right_config_sender_node = pipeline.create(dai.node.Script)
    right_config_sender_node.setScriptPath(
        Path(__file__).parent / "utils/config_sender_script.py"
    )
    right_config_sender_node.inputs["frame_input"].setBlocking(True)
    right_config_sender_node.inputs["config_input"].setBlocking(True)
    right_config_sender_node.inputs["frame_input"].setMaxSize(30)
    right_config_sender_node.inputs["config_input"].setMaxSize(30)

    input_node.link(right_config_sender_node.inputs["frame_input"])
    detection_process_node.right_config_output.link(
        right_config_sender_node.inputs["config_input"]
    )

    right_crop_node = pipeline.create(dai.node.ImageManipV2)
    right_crop_node.initialConfig.setReusePreviousImage(False)
    right_crop_node.inputConfig.setReusePreviousMessage(False)
    right_crop_node.inputImage.setReusePreviousMessage(False)

    right_config_sender_node.outputs["output_config"].link(right_crop_node.inputConfig)
    right_config_sender_node.outputs["output_frame"].link(right_crop_node.inputImage)

    face_config_sender_node = pipeline.create(dai.node.Script)
    face_config_sender_node.setScriptPath(
        Path(__file__).parent / "utils/config_sender_script.py"
    )
    face_config_sender_node.inputs["frame_input"].setBlocking(True)
    face_config_sender_node.inputs["config_input"].setBlocking(True)
    face_config_sender_node.inputs["frame_input"].setMaxSize(30)
    face_config_sender_node.inputs["config_input"].setMaxSize(30)

    input_node.link(face_config_sender_node.inputs["frame_input"])
    detection_process_node.face_config_output.link(
        face_config_sender_node.inputs["config_input"]
    )

    face_crop_node = pipeline.create(dai.node.ImageManipV2)
    face_crop_node.initialConfig.setReusePreviousImage(False)
    face_crop_node.inputConfig.setReusePreviousMessage(False)
    face_crop_node.inputImage.setReusePreviousMessage(False)

    face_config_sender_node.outputs["output_config"].link(face_crop_node.inputConfig)
    face_config_sender_node.outputs["output_frame"].link(face_crop_node.inputImage)

    # head_pose_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
    #     face_crop_node.out, "luxonis/head-pose-estimation:60x60")
    head_pose_node = pipeline.create(dai.node.NeuralNetwork)
    head_pose_node.setFromModelZoo(
        dai.NNModelDescription("luxonis/head-pose-estimation:60x60"), useCached=True
    )
    face_crop_node.out.link(head_pose_node.input)

    head_pose_script = pipeline.create(dai.node.Script)
    head_pose_script.setScriptPath(Path(__file__).parent / "utils/head_pose_script.py")
    head_pose_node.outp
    head_pose_node.getOutput(0).link(head_pose_script.inputs["yaw_input"])
    head_pose_node.getOutput(1).link(head_pose_script.inputs["pitch_input"])
    head_pose_node.getOutput(2).link(head_pose_script.inputs["roll_input"])

    # gaze_estimation_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(

    gaze_estimation_node = pipeline.create(dai.node.NeuralNetwork)
    gaze_estimation_node.setFromModelZoo(
        dai.NNModelDescription("luxonis/gaze-estimation-adas:60x60"), useCached=True
    )

    head_pose_script.outputs["head_pose_output"].link(
        gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"]
    )
    left_crop_node.out.link(gaze_estimation_node.inputs["left_eye_image"])
    right_crop_node.out.link(gaze_estimation_node.inputs["right_eye_image"])

    # age_gender_node: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
    #     left_crop_node.out, "luxonis/age-gender-recognition:62x62"
    # )

    # sync_node = pipeline.create(DetectionsAgeGenderSync)
    # input_node.link(sync_node.passthrough_input)
    # face_detection_node.out.link(sync_node.detections_input)
    # age_gender_node.getOutput(0).link(sync_node.age_input)
    # age_gender_node.getOutput(1).link(sync_node.gender_input)

    # annotation_node = pipeline.create(AnnotationNode)
    # sync_node.out.link(annotation_node.input)

    # visualizer.addTopic("Video", sync_node.out_frame)
    # visualizer.addTopic("Text", annotation_node.output)

    frame_q = face_detection_node.passthrough.createOutputQueue()
    det_q = face_detection_node.out.createOutputQueue()

    right_crop_q = right_crop_node.out.createOutputQueue()
    left_crop_q = left_crop_node.out.createOutputQueue()

    head_pose_q = head_pose_node.getOutput(0).createOutputQueue()  # yaw, pitch, roll
    # head_pose_q = head_pose_node.out.createOutputQueue()

    gaze_q = gaze_estimation_node.out.createOutputQueue()

    print("Pipeline created.")
    pipeline.start()
    # visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        # key = visualizer.waitKey(1)
        # if key == ord("q"):
        #     print("Got q key. Exiting...")
        #     break

        in_frame = frame_q.get()
        frame = in_frame.getCvFrame()
        w, h = frame.shape[1], frame.shape[0]

        left_eye = left_crop_q.get().getCvFrame()
        right_eye = right_crop_q.get().getCvFrame()

        detections = det_q.get().detections

        for detection in detections:
            corners = detection.rotated_rect.denormalize(w, h).getPoints()
            corners = [(int(pt.x), int(pt.y)) for pt in corners]
            cv2.polylines(frame, [np.array(corners)], True, (255, 0, 0), 2)

            keypoints = detection.keypoints
            keypoints = [(int(pt.x * w), int(pt.y * h)) for pt in keypoints]
            for i, pt in enumerate(keypoints):
                cv2.putText(
                    frame, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
                cv2.circle(frame, pt, 2, (0, 255, 0), 2)

        gaze = gaze_q.get()
        print(gaze)
        head_pose = head_pose_q.get()
        print(type(head_pose))
        for prediction in head_pose.predictions:
            print(prediction.prediction)

        cv2.imshow("left_eye", left_eye)
        cv2.imshow("right_eye", right_eye)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
