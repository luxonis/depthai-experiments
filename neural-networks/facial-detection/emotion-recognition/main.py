from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData, ImgDetectionsBridge
from depthai_nodes.node.utils import generate_script_content
from utils.arguments import initialize_argparser
from utils.host_process_detections import ProcessDetections
from utils.host_sync import DetectionSyncNode
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

DET_MODEL = "luxonis/yunet:640x480"
REC_MODEL = "luxonis/emotion-recognition:260x260"

REQ_WIDTH, REQ_HEIGHT = (
    1024,
    768,
)  # we are requesting larger input size than required because we want to keep some resolution for the second stage model

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # crowd counting model
    det_model_description = dai.NNModelDescription(DET_MODEL)
    det_model_description.platform = platform
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # emotion recognition model
    rec_model_description = dai.NNModelDescription(REC_MODEL)
    rec_model_description.platform = platform
    rec_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))

    # media/camera input
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        replay.setSize(REQ_WIDTH, REQ_HEIGHT)
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam = cam.requestOutput(
            size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=args.fps_limit
        )
    input_node = replay.out if args.media_path else cam

    # resize to det model input size
    resize_node = pipeline.create(dai.node.ImageManipV2)
    resize_node.initialConfig.setOutputSize(
        det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
    )
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)
    input_node.link(resize_node.inputImage)

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out, det_model_nn_archive
    )

    # detection processing
    det_bridge = pipeline.create(ImgDetectionsBridge).build(
        det_nn.out
    )  # TODO: remove once we have it working with ImgDetectionsExtended
    script_node = pipeline.create(dai.node.Script)
    det_bridge.out.link(script_node.inputs["det_in"])
    input_node.link(script_node.inputs["preview"])
    script_content = generate_script_content(
        resize_width=rec_model_nn_archive.getInputWidth(),
        resize_height=rec_model_nn_archive.getInputHeight(),
    )
    script_node.setScript(script_content)

    crop_node = pipeline.create(dai.node.ImageManipV2)
    crop_node.inputConfig.setWaitForMessage(True)

    script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
    script_node.outputs["manip_img"].link(crop_node.inputImage)

    emotion_recognition_node: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(crop_node.out, rec_model_nn_archive)

    # sync detections and recognitions
    sync_node = pipeline.create(DetectionSyncNode)
    input_node.link(sync_node.passthrough_input)
    det_nn.out.link(sync_node.detections_input)
    emotion_recognition_node.out.link(sync_node.emotion_input)

    # annotation
    annotation_node = pipeline.create(AnnotationNode)
    sync_node.out.link(annotation_node.input)

    # visualization
    visualizer.addTopic("Video", sync_node.out_frame)
    visualizer.addTopic("Emotions", annotation_node.output)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
