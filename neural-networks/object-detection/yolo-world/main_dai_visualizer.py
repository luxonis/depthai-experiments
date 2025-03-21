import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from depthai_nodes import ImgDetectionsExtended
import argparse
import os


from utils import download_model

MAX_NUM_CLASSES = 80
QUANT_ZERO_POINT = 89.0
QUANT_SCALE = 0.003838143777
IMAGE_SIZE = (640, 640)


class DetectionLabelCustomizer(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self._accepted_labels = []
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self, nn: dai.Node.Output, accepted_labels: list, class_names: list
    ) -> "DetectionLabelCustomizer":
        self._accepted_labels = accepted_labels
        self._class_names = class_names
        self.link_args(nn)
        return self

    def process(self, detections: dai.Buffer) -> None:
        assert isinstance(
            detections,
            (dai.ImgDetections, dai.SpatialImgDetections, ImgDetectionsExtended),
        )

        filtered_detections = [
            i for i in detections.detections if i.label in self._accepted_labels
        ]

        for det in filtered_detections:
            det.label_name = self._class_names[det.label]

        img_detections = type(detections)()
        img_detections.detections = filtered_detections
        img_detections.setTimestamp(detections.getTimestamp())
        img_detections.setSequenceNum(detections.getSequenceNum())
        self.output.send(img_detections)


def extract_text_embeddings(class_names):
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text = tokenizer(text=class_names, return_tensors="pt", padding=True)
    text_onnx = text["input_ids"].detach().cpu().numpy().astype(np.int64)
    attention_mask = (text_onnx != 0).astype(np.int64)

    textual_onnx_model_path = download_model(
        "https://huggingface.co/jmzzomg/clip-vit-base-patch32-text-onnx/resolve/main/model.onnx",
        "clip_textual_hf.onnx",
    )

    session_textual = onnxruntime.InferenceSession(
        textual_onnx_model_path,
        providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    textual_output = session_textual.run(
        None,
        {
            session_textual.get_inputs()[0].name: text_onnx,
            "attention_mask": attention_mask,
        },
    )[0]

    num_padding = MAX_NUM_CLASSES - len(class_names)
    text_features = np.pad(
        textual_output, ((0, num_padding), (0, 0)), mode="constant"
    ).T.reshape(1, 512, MAX_NUM_CLASSES)
    text_features = (text_features / QUANT_SCALE) + QUANT_ZERO_POINT
    text_features = text_features.astype("uint8")

    del session_textual

    return text_features


def main(args):
    text_features = extract_text_embeddings(args.class_names)
    device = (
        dai.Device(dai.DeviceInfo(args.device)) if args.device != "" else dai.Device()
    )

    visualizer = dai.RemoteConnection()
    with dai.Pipeline(device) as pipeline:
        manip = pipeline.create(dai.node.ImageManipV2)
        manip.setMaxOutputFrameSize(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3)
        manip.initialConfig.setOutputSize(
            IMAGE_SIZE[0], IMAGE_SIZE[1], dai.ImageManipConfigV2.ResizeMode.LETTERBOX
        )

        if args.video_path is not None:
            replayNode = pipeline.create(dai.node.ReplayVideo)
            replayNode.setOutFrameType(dai.ImgFrame.Type.BGR888i)
            replayNode.setReplayVideoFile(args.video_path)

            replayNode.out.link(manip.inputImage)
        else:
            cam = pipeline.create(dai.node.Camera).build(
                boardSocket=dai.CameraBoardSocket.CAM_A
            )
            camOut = cam.requestOutput(IMAGE_SIZE, dai.ImgFrame.Type.RGB888i)

            camOut.link(manip.inputImage)

        model_description = dai.NNModelDescription(
            modelSlug="yolo-world-l",
            modelVersionSlug="640x640-host-decoding",
            platform="RVC4",
        )
        archive_path = dai.getModelFromZoo(model_description, useCached=True)
        nn_archive = dai.NNArchive(archive_path)

        nn_with_parser = pipeline.create(ParsingNeuralNetwork)
        nn_with_parser.setNNArchive(nn_archive)
        nn_with_parser.setBackend("snpe")
        nn_with_parser.setBackendProperties(
            {"runtime": "dsp", "performance_profile": "default"}
        )
        nn_with_parser.setNumInferenceThreads(1)
        nn_with_parser.getParser(0).setConfidenceThreshold(args.confidence_threshold)

        manip.out.link(nn_with_parser.inputs["images"])

        textInputQueue = nn_with_parser.inputs["texts"].createInputQueue()
        nn_with_parser.inputs["texts"].setReusePreviousMessage(True)

        detection_label_filter = pipeline.create(DetectionLabelCustomizer)
        detection_label_filter.build(
            nn_with_parser.out, list(range(len(args.class_names))), args.class_names
        )

        visualizer.addTopic("Detections", detection_label_filter.output)
        visualizer.addTopic("Color", nn_with_parser.passthroughs["images"])

        pipeline.start()

        inputNNData = dai.NNData()
        inputNNData.addTensor(
            "texts", text_features, dataType=dai.TensorInfo.DataType.U8F
        )
        textInputQueue.send(inputNNData)

        print("Press 'q' to stop")
        while pipeline.isRunning():
            pipeline.processTasks()
            key = visualizer.waitKey(1)
            if key == ord("q"):
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Yolo World Demo")
    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default=None,
        help="The path to the video file",
    )
    parser.add_argument(
        "-c",
        "--class_names",
        type=str,
        nargs="+",
        required=True,
        help="Class names to be detected",
    )
    parser.add_argument(
        "-t",
        "--confidence_threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for detections",
    )
    return parser.parse_args()


def check_args(args):
    if len(args.class_names) > MAX_NUM_CLASSES:
        raise ValueError(
            f"Number of classes exceeds the maximum number of classes: {MAX_NUM_CLASSES}"
        )
    if args.video_path is not None and not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found at {args.video_path}")

    if args.video_path is None:
        print("No video file provided. Using camera input.")


if __name__ == "__main__":
    args = parse_args()
    check_args(args)

    main(args)
