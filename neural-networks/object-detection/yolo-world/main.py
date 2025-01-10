import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import cv2
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
import argparse
import os


from utils import draw_detections, download_model

MAX_NUM_CLASSES = 80
QUANT_ZERO_POINT = 89.0
QUANT_SCALE = 0.003838143777
IMAGE_SIZE = (640, 640)


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
            model="yolo-world-l", platform="RVC4"
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
        nn_with_parser.getParser(0).setInputImageSize(IMAGE_SIZE[0], IMAGE_SIZE[1])

        manip.out.link(nn_with_parser.inputs["images"])
        qDet = nn_with_parser.out.createOutputQueue()
        qImg = nn_with_parser.passthroughs["images"].createOutputQueue()

        textInputQueue = nn_with_parser.inputs["texts"].createInputQueue()
        nn_with_parser.inputs["texts"].setReusePreviousMessage(True)

        pipeline.start()

        inputNNData = dai.NNData()
        inputNNData.addTensor(
            "texts", text_features, dataType=dai.TensorInfo.DataType.U8F
        )
        textInputQueue.send(inputNNData)

        print("Press 'q' to stop")
        while pipeline.isRunning():
            inDet: dai.ImgDetections = qDet.get()
            inImage: dai.ImgFrame = qImg.get()
            cvFrame = inImage.getCvFrame()
            visFrame = draw_detections(cvFrame, inDet.detections, args.class_names)
            cv2.imshow("Inference result", visFrame)
            key = cv2.waitKey(1)
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
