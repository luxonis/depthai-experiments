import argparse
from pathlib import Path
import json
import depthai as dai
from car_detection import CarDetection
from video import Video

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model path for inference",
                    default='models/yolo_v4_tiny_openvino_2021.3_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='yolo-tiny.json', type=str)
parser.add_argument("-vid", "--vid", help="Provide Luxonis video name or path to video that should be used instead of camera",
                    default="cars-tracking-above-01", type=str)
parser.add_argument("-cam", "--cam", help="Use camera instead of video.", 
                    action="store_true")
args = parser.parse_args()


cam = args.cam
if not cam:
    video = args.vid
    video_path = Video(video).get_path()

with dai.Pipeline() as pipeline:
    nn_input_shape = (512, 320)

    if cam:
        cam = pipeline.create(dai.node.ColorCamera).build()
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setPreviewSize(nn_input_shape)
        cam.setInterleaved(False)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(video_path)
        replay.setLoop(False)
        replay.setSize(nn_input_shape)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
    
    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    nn.setBlobPath(args.model)
    
    with open(str(Path(args.config).resolve())) as file:
        conf: dict = json.load(file)['nn_config']['NN_specific_metadata']
        nn.setNumClasses(conf['classes'])
        nn.setCoordinateSize(conf['coordinates'])
        nn.setAnchors(conf['anchors'])
        nn.setAnchorMasks(conf['anchor_masks'])
        nn.setIouThreshold(conf['iou_threshold'])

        if conf['confidence_threshold'] is not None:
            nn.setConfidenceThreshold(conf['confidence_threshold'])

    if cam:
        output = cam.preview
        cam.preview.link(nn.input)
    else:
        output = replay.out
        replay.out.link(nn.input)

    pipeline.create(CarDetection).build(output, nn.out)
    pipeline.run()