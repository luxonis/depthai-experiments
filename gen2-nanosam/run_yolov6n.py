import depthai as dai
import cv2
import numpy as np
import argparse
import blobconverter

from decoder import ONNXDecoder
from utils import generate_overlay, frame_norm, resize_and_pad

# --- constants ---
YOLO_WIDTH, YOLO_HEIGHT = 512, 288
NN_WIDTH, NN_HEIGHT = 1024, 1024

# --- arguments ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video file to be used for inference (conflicts with -cam).",
)
#parser.add_argument(
#    "-enc", "--encoder", type=str, help="Path to encoder blob that can run on device."
#)
parser.add_argument(
    "-dec", "--decoder", type=str, help="Path to decoder onnx that runs on host."
)
args = parser.parse_args()

is_camera = args.video is None

# --- pipeline ---

# [cam / input] --> [neural_network] --> [output]
pipeline = dai.Pipeline()

# create neural network node
embedding_nn = pipeline.create(dai.node.NeuralNetwork)
encoderPath = str(blobconverter.from_zoo("nanosam_resnet18_image_encoder_1024x1024", shaves = 6, zoo_type = "depthai", version="2022.1", use_cache=True))
embedding_nn.setBlobPath(encoderPath)
embedding_nn.setNumInferenceThreads(1)

detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
yoloPath = str(blobconverter.from_zoo("yolov6n_coco_416x416", shaves = 6, zoo_type = "depthai", use_cache=True))
detection_nn.setBlobPath(yoloPath)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setAnchors([])
detection_nn.setAnchorMasks({})
detection_nn.setIouThreshold(0.45)
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumInferenceThreads(1)

manip = pipeline.create(dai.node.ImageManip)
manip.setResize(416, 416)
manip.out.link(detection_nn.input)
manip.setWaitForConfigInput(False)

if is_camera:
    # create camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setInterleaved(False)

    # link
    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("rgb")
    cam_rgb.preview.link(cam_xout.input)
    cam_rgb.preview.link(embedding_nn.input)
    cam_rgb.preview.link(manip.inputImage)
else:
    # input node
    rgb_in = pipeline.create(dai.node.XLinkIn)
    rgb_in.setStreamName("in_nn")

    # link
    rgb_in.out.link(embedding_nn.input)
    rgb_in.out.link(manip.inputImage)




# outputs
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_det = pipeline.create(dai.node.XLinkOut)
xout_det.setStreamName("det")

# link
detection_nn.out.link(xout_det.input)
embedding_nn.out.link(xout_nn.input)


# --- helpers ---
POINTS = []
POINT_LABELS = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_MBUTTONDOWN:
        POINTS.append([x, y])
        POINT_LABELS.append(0)


# --- inference ---

decoder = ONNXDecoder(onnx_path=args.decoder)

# init CV2
cv2.imshow("RGB", np.zeros((NN_HEIGHT, NN_WIDTH, 3), dtype=np.uint8))
cv2.setMouseCallback("RGB", click_event)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if is_camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=3, blocking=True)
    else:
        video_rec = cv2.VideoCapture(args.video)
        q_in = device.getInputQueue("in_nn")
    q_nn = device.getOutputQueue(name="nn", maxSize=3, blocking=True)
    q_det = device.getOutputQueue(name="det", maxSize=3, blocking=True)


    def get_frame():
        if is_camera:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            return True, frame
        else:
            ok, frame = video_rec.read()
            if not ok:
                return ok, frame

            frame = resize_and_pad(frame, (NN_WIDTH, NN_HEIGHT))
            return ok, frame

    while video_rec.isOpened() if not is_camera else True:
        read_correctly, frame = get_frame()
        if not read_correctly:
            break

        if not is_camera:
            img = dai.ImgFrame()
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setData(cv2.resize(frame, (NN_WIDTH, NN_HEIGHT)).transpose(2, 0, 1).flatten())
            img.setWidth(NN_WIDTH)
            img.setHeight(NN_HEIGHT)
            q_in.send(img)

        nn_out = q_nn.get()
        embeddings = nn_out.getFirstLayerFp16()
        embeddings = np.array(embeddings).reshape(1, 256, 64, 64)

        nn_det = q_det.get()
        detections = nn_det.detections
        for detection in detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            
            # prepare points
            points = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[3]],
                *POINTS
            ])
            point_labels = np.array([2, 3, *POINT_LABELS])

            # draw bbox
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,255), 2)

            mask, _, _ = decoder.predict(embeddings, points, point_labels)
            mask = (mask[:, :, 0] > 0).astype(np.uint8)

            # overlay mask
            frame = generate_overlay(frame, mask)

        # visualize keypoints
        for i, p in enumerate(POINTS):
            x, y = p
            c = (0, 255, 0) if POINT_LABELS[i] == 1 else (0, 0, 255)
            cv2.circle(
                frame, (x, y), 5, c, -1
            )  # Draws a small green circle where the click occurred

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) == ord("q"):
            break
