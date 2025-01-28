from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork

from utils.arguments import initialize_argparser
from utils.process import ProcessDetections
from utils.sync import AnnotationSyncNode

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device_id)) if args.device_id else dai.Device()
platform = device.getPlatform().name
frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

if platform == "RVC2":
    raise NotImplementedError("RVC2 is not supported yet.")

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # Detection Model NN Archive
    det_model_description = dai.NNModelDescription(args.det_model)
    det_model_description.platform = platform
    det_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))
    det_nn_width = det_nn_archive.getInputWidth()
    det_nn_height = det_nn_archive.getInputHeight()
    det_nn_stride = (
        (det_nn_width + 7) // 8 * 8
    )  # Align width up to the nearest multiple of 8

    # Recognition Model NN Archive
    rec_model_description = dai.NNModelDescription(args.rec_model)
    rec_model_description.platform = platform
    rec_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))

    # Video/Camera Input Node
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        imageManip = pipeline.create(dai.node.ImageManipV2)
        imageManip.setMaxOutputFrameSize(det_nn_stride * det_nn_height * 3)
        imageManip.initialConfig.setOutputSize(det_nn_width, det_nn_height)
        imageManip.initialConfig.setFrameType(frame_type)
        replay.out.link(imageManip.inputImage)
    else:
        cam = pipeline.create(dai.node.Camera).build()
    input_node = imageManip.out if args.media_path else cam

    # Detection Model Node
    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_nn_archive, fps=args.fps_limit
    )

    # Detections Processing Node
    det_process_node = pipeline.create(ProcessDetections)
    det_process_node.set_target_size(
        rec_nn_archive.getInputWidth(), rec_nn_archive.getInputHeight()
    )
    det_nn.out.link(det_process_node.detections_input)

    # Crop Configuration Sender Node
    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(
        str(Path(__file__).parent / "utils/config_sender_script.py")
    )
    config_sender_node.inputs["frame_input"].setMaxSize(30)
    config_sender_node.inputs["config_input"].setMaxSize(30)
    config_sender_node.inputs["num_configs_input"].setMaxSize(30)

    det_nn.passthrough.link(config_sender_node.inputs["frame_input"])
    det_process_node.config_output.link(config_sender_node.inputs["config_input"])
    det_process_node.num_configs_output.link(
        config_sender_node.inputs["num_configs_input"]
    )

    # Crop Node
    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setReusePreviousImage(False)
    crop_node.inputConfig.setReusePreviousMessage(False)
    crop_node.inputImage.setReusePreviousMessage(False)
    crop_node.inputConfig.setMaxSize(30)
    crop_node.inputImage.setMaxSize(30)
    crop_node.setNumFramesPool(30)
    crop_node.inputConfig.setWaitForMessage(True)

    config_sender_node.outputs["output_config"].link(crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(crop_node.inputImage)

    # Recognition Model Node
    rec_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, rec_nn_archive
    )

    # Annotation Sync Node
    annotation_sync_node = pipeline.create(
        AnnotationSyncNode, csim=args.cos_similarity_threshold
    )
    det_nn.out.link(annotation_sync_node.input_detections)
    rec_nn.out.link(annotation_sync_node.input_recognitions)

    # Visualizer
    visualizer.addTopic("Video", det_nn.passthrough, "images")
    visualizer.addTopic("Objects", annotation_sync_node.output_detections, "images")

    print("Pipeline created.")

    # Start Pipeline
    pipeline.start()

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
