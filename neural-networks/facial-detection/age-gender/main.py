from pathlib import Path
import depthai as dai

from depthai_nodes.node import ParsingNeuralNetwork
from depthai_nodes.node import GatherData

from utils.arguments import initialize_argparser
from utils.host_process_detections import ProcessDetections
from utils.annotation_node import AnnotationNode


_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()

DET_MODEL = "luxonis/yunet:640x480"
REC_MODEL = "luxonis/age-gender-recognition:62x62"

fps = args.fps_limit


def return_one(reference):
    return 1


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # face detection model
    det_model_description = dai.NNModelDescription(DET_MODEL)
    det_model_description.platform = platform
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # age-gender recognition model
    rec_model_description = dai.NNModelDescription(REC_MODEL)
    rec_model_description.platform = platform
    rec_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))

    # media/camera source
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(
            dai.ImgFrame.Type.BGR888i
            if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        replay.setSize(
            det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
        )
    input_node = (
        replay.out if args.media_path else pipeline.create(dai.node.Camera).build()
    )

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_model_nn_archive, fps=args.fps_limit
    )

    # detection processing
    detection_process_node = pipeline.create(ProcessDetections)
    detection_process_node.set_source_size(
        det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
    )
    detection_process_node.set_target_size(
        rec_model_nn_archive.getInputWidth(),
        rec_model_nn_archive.getInputHeight(),
    )
    det_nn.out.link(detection_process_node.detections_input)

    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(
        Path(__file__).parent
        / "utils/config_sender_script.py"  # TODO: utilize generate_script_content
    )
    config_sender_node.inputs["frame_input"].setBlocking(True)
    config_sender_node.inputs["config_input"].setBlocking(True)
    config_sender_node.inputs["frame_input"].setMaxSize(30)
    config_sender_node.inputs["config_input"].setMaxSize(30)

    input_node.link(config_sender_node.inputs["frame_input"])
    detection_process_node.config_output.link(config_sender_node.inputs["config_input"])

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(False)

    config_sender_node.outputs["output_config"].link(crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(crop_node.inputImage)

    rec_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, rec_model_nn_archive
    )

    # recognitions sync
    gather_rec = pipeline.create(GatherData).build(fps, wait_count_fn=return_one)
    rec_nn.getOutput(0).link(gather_rec.input_data)  # gender
    rec_nn.getOutput(1).link(gather_rec.input_reference)  # age

    # detections and recognitions sync
    gather_data_node = pipeline.create(GatherData).build(fps)
    gather_rec.out.link(gather_data_node.input_data)
    det_nn.out.link(gather_data_node.input_reference)

    # annotation
    annotation_node = pipeline.create(AnnotationNode).build(gather_data_node.out)

    # visualization
    visualizer.addTopic("Video", det_nn.passthrough, "images")
    visualizer.addTopic("AgeGender", annotation_node.out, "images")

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
