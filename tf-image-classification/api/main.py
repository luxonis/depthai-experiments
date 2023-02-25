#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug
camera = not args.video

# Start defining a pipeline
pipeline = dai.Pipeline()

# NeuralNetwork
print("Creating Neural Network...")
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(str(Path("flower.blob").resolve().absolute()))

if camera:
    print("Creating Color Camera...")
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(480, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("rgb")
    cam_rgb.preview.link(cam_xout.input)

    print("Creating ImageManip node...")
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(180, 180)
    cam_rgb.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

else:
    face_in = pipeline.create(dai.node.XLinkIn)
    face_in.setStreamName("in_nn")
    face_in.out.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

frame = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(in_frame, bbox):
    norm_vals = np.full(len(bbox), in_frame.shape[0])
    norm_vals[::2] = in_frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    else:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        detection_in = device.getInputQueue("in_nn")
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        if camera:
            in_rgb = q_rgb.get()
            new_frame = np.array(in_rgb.getData()).reshape((3, in_rgb.getHeight(), in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8)
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            return True, np.ascontiguousarray(new_frame)
        else:
            return cap.read()


    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    result = None

    while should_run():
        read_correctly, frame = get_frame()

        if not read_correctly:
            break

        if not camera:
            nn_data = dai.NNData()
            nn_data.setLayer("input", to_planar(frame, (180, 180)))
            detection_in.send(nn_data)

        in_nn = q_nn.tryGet()

        if in_nn is not None:
            data = softmax(in_nn.getFirstLayerFp16())
            result_conf = np.max(data)
            if result_conf > 0.5:
                result = {
                    "name": class_names[np.argmax(data)],
                    "conf": round(100 * result_conf, 2)
                }
            else:
                result = None

        if debug:
            # if the frame is available, draw bounding boxes on it and show the frame
            if result is not None:
                cv2.putText(frame, "{} ({}%)".format(result["name"], result["conf"]), (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        elif result is not None:
            print("{} ({}%)".format(result["name"], result["conf"]))
