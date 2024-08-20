#!/usr/bin/env python3
import depthai as dai
import argparse
from host_decoding import HostDecoding

'''
YoloV5 object detector running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob is taken from ML training examples:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks

You can clone the YoloV5_training.ipynb notebook and try training the model yourself.

'''
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]
cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/yolov5s_default_openvino_2021.4_6shave.blob', type=str)
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.3, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.4, type=float)


args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model
conf_thresh = args.confidence_thresh
iou_thresh = args.iou_thresh

NN_SHAPE = 416

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(nn_path)

    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    cam=None
    if cam_source == 'rgb':
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(NN_SHAPE, NN_SHAPE)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.preview.link(detection_nn.input)
    elif cam_source == 'left':
        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    elif cam_source == 'right':
        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    if cam_source != 'rgb':
        manip = pipeline.create(dai.node.ImageManip)
        manip.setResize(NN_SHAPE,NN_SHAPE)
        manip.setKeepAspectRatio(True)
        manip.setFrameType(dai.ImgFrame.Type.BGR888p)
        cam.out.link(manip.inputImage)
        manip.out.link(detection_nn.input)

    cam.setFps(40)
    host_decoding = pipeline.create(HostDecoding).build(
        img_output=cam.preview,
        nn_path=nn_path, 
        nn_data=detection_nn.out,
        label_map=labelMap
        )
    
    host_decoding.set_conf_thresh(conf_thresh)
    host_decoding.set_iou_thresh(iou_thresh)
    
    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")
