import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import cv2
import depthai as dai
import argparse
import requests
import os

from utils import draw_detections, download_model

MAX_NUM_CLASSES = 80
QUANT_ZERO_POINT = 89.0
QUANT_SCALE = 0.003838143777


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
    device_info = dai.DeviceInfo(args.device_ip)
    device = dai.Device(device_info)

    with dai.Pipeline(device) as pipeline:

        manip = pipeline.create(dai.node.ImageManipV2)
        manip.setMaxOutputFrameSize(640 * 640 * 3)
        manip.initialConfig.setOutputSize(
            640, 640, dai.ImageManipConfigV2.ResizeMode.LETTERBOX
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
            camOut = cam.requestOutput((640, 640), dai.ImgFrame.Type.RGB888i)

            camOut.link(manip.inputImage)

        model_description = dai.NNModelDescription(
            model="yolo-world-l", platform="RVC4"
        )
        archive_path = dai.getModelFromZoo(model_description, useCached=True)
        nn_archive = dai.NNArchive(archive_path)

        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setNNArchive(nn_archive)

        nn.setBackend("snpe")
        nn.setBackendProperties({"runtime": "dsp", "performance_profile": "default"})

        detectionParser = pipeline.create(dai.node.DetectionParser)
        detectionParser.setConfidenceThreshold(args.confidence_threshold)
        detectionParser.setNumClasses(MAX_NUM_CLASSES)
        detectionParser.setCoordinateSize(4)
        detectionParser.setIouThreshold(0.3)
        detectionParser.setInputImageSize(640, 640)
        nn.setNumInferenceThreads(1)

        # Linking
        manip.out.link(nn.inputs["images"])
        qDet = detectionParser.out.createOutputQueue()
        # qImg = nn.passthroughs['images'].createOutputQueue()
        qImg = manip.out.createOutputQueue()
        nn.out.link(detectionParser.input)

        textInputQueue = nn.inputs["texts"].createInputQueue()
        nn.inputs["texts"].setReusePreviousMessage(True)

        pipeline.start()

        inputNNData = dai.NNData()
        inputNNData.addTensor(
            "texts", text_features, dataType=dai.TensorInfo.DataType.U8F
        )
        textInputQueue.send(inputNNData)

        print("Press 'q' to stop")
        while pipeline.isRunning():  # and replayNode.isRunning():
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
        "--device_ip",
        type=str,
        required=True,
        help="The IP address of the target device",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path to the video file",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        required=True,
        help="Class names to be detected",
    )
    parser.add_argument(
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
