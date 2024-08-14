import argparse
import depthai as dai
from car_detection import CarDetection
from video import Video

modelDescription = dai.NNModelDescription(modelSlug="yolov6-nano", platform="RVC2")
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)


parser = argparse.ArgumentParser()
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
    nn_input_shape = (512, 288)

    if cam:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setPreviewSize(nn_input_shape)
        cam.setInterleaved(False)
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(video_path)
        replay.setLoop(False)
        replay.setSize(nn_input_shape)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
    
    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setNNArchive(nn_archive)

    if cam:
        output = cam.preview
        cam.preview.link(nn.input)
    else:
        output = replay.out
        replay.out.link(nn.input)

    pipeline.create(CarDetection).build(
        img_frame=output, 
        detections=nn.out,
        )
    
    print("Pipeline created")
    pipeline.run()
    print("Pipeline finished")
