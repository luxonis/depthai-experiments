# coding=utf-8
from pathlib import Path
import blobconverter
import depthai as dai
from gaze_estimation.detections_recognitions_sync import DetectionsRecognitionsSync
from gaze_estimation.gaze_estimation import GazeEstimation
from gaze_estimation.display import Display
from gaze_position_estimation import GazePositionEstimation


VIDEO_SIZE = (1072, 1072)
print("Creating pipeline...")
with (dai.Pipeline() as pipeline):
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
    openvino_version = '2021.4'

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1072, 1072)
    cam.setVideoSize(VIDEO_SIZE)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setPreviewNumFramesPool(20)
    cam.setFps(20)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    # Creating left/right mono cameras for StereoDepth
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setFps(15)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setFps(15)

    # Create StereoDepth node that will produce the depth map
    stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth).build(left.out, right.out)
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setSubpixelFractionalBits(3)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.setMaxOutputFrameSize(300*300*3)
    cam.preview.link(face_det_manip.inputImage)

    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6,
        version=openvino_version
    ))
    face_det_manip.out.link(face_det_nn.input)

    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)

    face_det_nn.out.link(script.inputs['face_det_in'])
    face_det_nn.passthrough.link(script.inputs['face_pass'])

    cam.preview.link(script.inputs['preview'])

    with open(Path(__file__).parent / "gaze_estimation" / "script.py", "r") as f:
        script.setScript(f.read())

    headpose_manip = pipeline.create(dai.node.ImageManip)
    headpose_manip.initialConfig.setResize(60, 60)
    script.outputs['headpose_cfg'].link(headpose_manip.inputConfig)
    script.outputs['headpose_img'].link(headpose_manip.inputImage)

    headpose_nn = pipeline.create(dai.node.NeuralNetwork)
    headpose_nn.setBlobPath(blobconverter.from_zoo(
        name="head-pose-estimation-adas-0001",
        shaves=6,
        version=openvino_version
    ))
    headpose_manip.out.link(headpose_nn.input)

    headpose_nn.out.link(script.inputs['headpose_in'])
    headpose_nn.passthrough.link(script.inputs['headpose_pass'])

    landmark_manip = pipeline.create(dai.node.ImageManip)
    landmark_manip.initialConfig.setResize(48, 48)
    script.outputs['landmark_cfg'].link(landmark_manip.inputConfig)
    script.outputs['landmark_img'].link(landmark_manip.inputImage)

    landmark_nn = pipeline.create(dai.node.NeuralNetwork)
    landmark_nn.setBlobPath(blobconverter.from_zoo(
        name="landmarks-regression-retail-0009",
        shaves=6,
        version=openvino_version
    ))
    landmark_manip.out.link(landmark_nn.input)

    landmark_nn.out.link(script.inputs['landmark_in'])
    landmark_nn.passthrough.link(script.inputs['landmark_pass'])

    left_manip = pipeline.create(dai.node.ImageManip)
    left_manip.initialConfig.setResize(60, 60)
    left_manip.inputConfig.setWaitForMessage(True)
    script.outputs['left_manip_img'].link(left_manip.inputImage)
    script.outputs['left_manip_cfg'].link(left_manip.inputConfig)
    left_manip.out.link(script.inputs['left_eye_in'])

    right_manip = pipeline.create(dai.node.ImageManip)
    right_manip.initialConfig.setResize(60, 60)
    right_manip.inputConfig.setWaitForMessage(True)
    script.outputs['right_manip_img'].link(right_manip.inputImage)
    script.outputs['right_manip_cfg'].link(right_manip.inputConfig)
    right_manip.out.link(script.inputs['right_eye_in'])

    gaze_nn = pipeline.create(dai.node.NeuralNetwork)
    gaze_nn.setBlobPath(blobconverter.from_zoo(
        name="gaze-estimation-adas-0002",
        shaves=6,
        version=openvino_version,
        compile_params=['-iop head_pose_angles:FP16,right_eye_image:U8,left_eye_image:U8']
    ))

    SCRIPT_OUTPUT_NAMES = ['to_gaze_head', 'to_gaze_left', 'to_gaze_right']
    NN_NAMES = ['head_pose_angles', 'left_eye_image', 'right_eye_image']
    for script_name, nn_name in zip(SCRIPT_OUTPUT_NAMES, NN_NAMES):
        script.outputs[script_name].link(gaze_nn.inputs[nn_name])
        gaze_nn.inputs[nn_name].setBlocking(True)
        gaze_nn.inputs[nn_name].setReusePreviousMessage(False)

    landmarks_sync = pipeline.create(DetectionsRecognitionsSync).build()
    face_det_nn.out.link(landmarks_sync.input_detections)
    landmark_nn.out.link(landmarks_sync.input_recognitions)

    gaze_sync = pipeline.create(DetectionsRecognitionsSync).build()
    face_det_nn.out.link(gaze_sync.input_detections)
    gaze_nn.out.link(gaze_sync.input_recognitions)

    gaze_estimation = pipeline.create(GazeEstimation).build(cam.video, gaze_sync.output, landmarks_sync.output)

    head_position_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork) # TODO replace with NN suggested on Slack
    cam.preview.link(head_position_nn.input)
    stereo.depth.link(head_position_nn.inputDepth)

    gaze_position_estimation = pipeline.create(GazePositionEstimation).build(
        img_frames=gaze_estimation.frame_output, gaze_detection_data=gaze_estimation.gaze_data_output, head_position_det=head_position_nn.out,
        head_position_depth=head_position_nn.passthroughDepth, head_rotation_nn_data=headpose_nn.out
    )

    print("Running pipeline...")
    pipeline.run()
    print("Pipeline exited.")