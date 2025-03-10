import depthai as dai
from pathlib import Path


def create_crop_node(
    pipeline: dai.Pipeline,
    input_frame: dai.Node.Output,
    configs_message: dai.Node.Output,
) -> dai.node.ImageManipV2:
    script_path = Path(__file__).parent / "config_sender_script.py"
    with script_path.open("r") as script_file:
        script_content = script_file.read()

    config_sender_script = pipeline.create(dai.node.Script)
    config_sender_script.setScript(script_content)
    config_sender_script.inputs["frame_input"].setBlocking(True)
    config_sender_script.inputs["config_input"].setBlocking(True)

    img_manip_node = pipeline.create(dai.node.ImageManipV2)
    img_manip_node.initialConfig.setReusePreviousImage(False)
    img_manip_node.inputConfig.setReusePreviousMessage(False)
    img_manip_node.inputImage.setReusePreviousMessage(False)
    img_manip_node.inputConfig.setBlocking(True)
    img_manip_node.inputImage.setBlocking(True)

    input_frame.link(config_sender_script.inputs["frame_input"])
    configs_message.link(config_sender_script.inputs["config_input"])

    config_sender_script.outputs["output_config"].link(img_manip_node.inputConfig)
    config_sender_script.outputs["output_frame"].link(img_manip_node.inputImage)

    return img_manip_node


def create_gaze_estimation_model(
    pipeline: dai.Pipeline,
    head_pose: dai.Node.Output,
    left_eye: dai.Node.Output,
    right_eye: dai.Node.Output,
) -> dai.node.NeuralNetwork:
    gaze_estimation_node = pipeline.create(dai.node.NeuralNetwork)
    gaze_estimation_node.setFromModelZoo(
        dai.NNModelDescription("luxonis/gaze-estimation-adas:60x60"), useCached=True
    )
    head_pose.link(gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"])
    left_eye.link(gaze_estimation_node.inputs["left_eye_image"])
    right_eye.link(gaze_estimation_node.inputs["right_eye_image"])
    gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"].setBlocking(True)
    gaze_estimation_node.inputs["left_eye_image"].setBlocking(True)
    gaze_estimation_node.inputs["right_eye_image"].setBlocking(True)
    gaze_estimation_node.inputs["left_eye_image"].setMaxSize(5)
    gaze_estimation_node.inputs["right_eye_image"].setMaxSize(5)
    gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"].setMaxSize(5)

    return gaze_estimation_node
