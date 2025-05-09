from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

from utils.arguments import initialize_argparser
from utils.process import ProcessDetections
from utils.sync import AnnotationSyncNode

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # detection model
    det_model_description = dai.NNModelDescription(args.det_model)
    det_model_description.platform = platform
    det_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # recognition model
    rec_model_description = dai.NNModelDescription(args.rec_model)
    rec_model_description.platform = platform
    rec_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))

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
        replay.setSize(det_nn_archive.getInputWidth(), det_nn_archive.getInputHeight())
    else:
        cam = pipeline.create(dai.node.Camera).build()
    input_node = replay if args.media_path else cam

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_nn_archive, fps=args.fps_limit
    )

    # detection processing
    det_bridge = pipeline.create(ImgDetectionsBridge).build(
        det_nn.out
    )  # TODO: remove once we have it working with ImgDetectionsExtended
    script_node = pipeline.create(dai.node.Script)
    det_bridge.out.link(script_node.inputs["det_in"])
    det_nn.passthrough.link(script_node.inputs["preview"])
    script_content = generate_script_content(
        resize_width=rec_nn_archive.getInputWidth(),
        resize_height=rec_nn_archive.getInputHeight(),
    )
    script_node.setScript(script_content)

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setOutputSize(
        rec_nn_archive.getInputWidth(), rec_nn_archive.getInputHeight()
    )
    crop_node.inputConfig.setWaitForMessage(True)

    script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
    script_node.outputs["manip_img"].link(crop_node.inputImage)

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
