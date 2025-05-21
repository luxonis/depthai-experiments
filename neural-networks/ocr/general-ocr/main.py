from pathlib import Path

import depthai as dai
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData, ImgDetectionsBridge
from depthai_nodes.node.utils import generate_script_content

from utils.annotation_node import OCRAnnotationNode
from utils.arguments import initialize_argparser

DET_MODEL = "luxonis/paddle-text-detection:320x576"
REC_MODEL = "luxonis/paddle-text-recognition:320x48"
REQ_WIDTH, REQ_HEIGHT = (
    1152,
    640,
)  # we are requesting larger input size than required because we want to keep some resolution for the second stage model

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

if args.fps_limit is None:
    args.fps_limit = 5 if platform == "RVC2" else 5
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # text detection model
    det_model_description = dai.NNModelDescription(DET_MODEL, platform=platform)
    det_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(det_model_description, useCached=False)
    )
    det_model_w, det_model_h = det_model_nn_archive.getInputSize()

    # text recognition model
    rec_model_description = dai.NNModelDescription(REC_MODEL, platform=platform)
    rec_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(rec_model_description, useCached=False)
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        replay.setSize(REQ_WIDTH, REQ_HEIGHT)
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=args.fps_limit
        )
    input_node_out = replay.out if args.media_path else cam_out

    # resize to det model input size
    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(det_model_w, det_model_h)
    resize_node.initialConfig.setReusePreviousImage(False)
    input_node_out.link(resize_node.inputImage)

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out, det_model_nn_archive
    )
    det_nn.setNumPoolFrames(30)

    # detection processing
    det_bridge = pipeline.create(ImgDetectionsBridge).build(
        det_nn.out, ignore_angle=True
    )  # TODO: remove once we have it working with ImgDetectionsExtended
    script_node = pipeline.create(dai.node.Script)
    det_bridge.out.link(script_node.inputs["det_in"])
    input_node_out.link(script_node.inputs["preview"])
    script_content = generate_script_content(
        resize_width=rec_model_nn_archive.getInputWidth(),
        resize_height=rec_model_nn_archive.getInputHeight(),
    )
    script_node.setScript(script_content)

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.initialConfig.setOutputSize(
        rec_model_nn_archive.getInputWidth(), rec_model_nn_archive.getInputHeight()
    )
    crop_node.inputConfig.setWaitForMessage(True)

    script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
    script_node.outputs["manip_img"].link(crop_node.inputImage)

    ocr_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, rec_model_nn_archive
    )
    ocr_nn.setNumPoolFrames(30)
    ocr_nn.input.setMaxSize(30)

    # detections and recognitions sync
    gather_data_node = pipeline.create(GatherData).build(args.fps_limit)
    ocr_nn.out.link(gather_data_node.input_data)
    det_nn.out.link(gather_data_node.input_reference)

    # annotation
    annotation_node = pipeline.create(OCRAnnotationNode)
    gather_data_node.out.link(annotation_node.input)
    det_nn.passthrough.link(annotation_node.passthrough)

    # visualization
    visualizer.addTopic("Video", annotation_node.frame_output)
    visualizer.addTopic("Text", annotation_node.text_annotations_output)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
