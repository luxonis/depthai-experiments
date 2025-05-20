import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsFilter,
    ImgDetectionsBridge,
)

from utils.helper_functions import extract_text_embeddings
from utils.arguments import initialize_argparser

MAX_NUM_CLASSES = 80
IMAGE_SIZE = (640, 640)


_, args = initialize_argparser()
if len(args.class_names) > MAX_NUM_CLASSES:
    raise ValueError(
        f"Number of classes exceeds the maximum number of classes: {MAX_NUM_CLASSES}"
    )

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()

text_features = extract_text_embeddings(
    class_names=args.class_names, max_num_classes=MAX_NUM_CLASSES
)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    manip = pipeline.create(dai.node.ImageManipV2)
    manip.setMaxOutputFrameSize(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3)
    manip.initialConfig.setOutputSize(
        IMAGE_SIZE[0], IMAGE_SIZE[1], dai.ImageManipConfigV2.ResizeMode.LETTERBOX
    )

    if args.media_path is not None:
        replayNode = pipeline.create(dai.node.ReplayVideo)
        replayNode.setOutFrameType(dai.ImgFrame.Type.BGR888i)
        replayNode.setReplayVideoFile(args.media_path)

        replayNode.out.link(manip.inputImage)
    else:
        cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        camOut = cam.requestOutput(IMAGE_SIZE, dai.ImgFrame.Type.RGB888i)

        camOut.link(manip.inputImage)

    model_description = dai.NNModelDescription(
        model="yolo-world-l:640x640-host-decoding",
        platform="RVC4",
    )

    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description, useCached=True))

    nn_with_parser = pipeline.create(ParsingNeuralNetwork)
    nn_with_parser.setNNArchive(nn_archive)
    nn_with_parser.setBackend("snpe")
    nn_with_parser.setBackendProperties(
        {"runtime": "dsp", "performance_profile": "default"}
    )
    nn_with_parser.setNumInferenceThreads(1)
    nn_with_parser.getParser(0).setConfidenceThreshold(args.confidence_thresh)

    manip.out.link(nn_with_parser.inputs["images"])

    textInputQueue = nn_with_parser.inputs["texts"].createInputQueue()
    nn_with_parser.inputs["texts"].setReusePreviousMessage(True)

    # filter and rename detection labels
    det_process_filter = pipeline.create(ImgDetectionsFilter).build(nn_with_parser.out)
    det_process_filter.setLabels(
        labels=[i for i in range(len(args.class_names))], keep=True
    )
    det_process_bridge = pipeline.create(ImgDetectionsBridge).build(
        det_process_filter.out,
        label_encoding={k: v for k, v in enumerate(args.class_names)},
    )

    visualizer.addTopic("Detections", det_process_bridge.out)
    visualizer.addTopic("Video", nn_with_parser.passthroughs["images"])

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    inputNNData = dai.NNData()
    inputNNData.addTensor("texts", text_features, dataType=dai.TensorInfo.DataType.U8F)
    textInputQueue.send(inputNNData)

    print("Press 'q' to stop")

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
