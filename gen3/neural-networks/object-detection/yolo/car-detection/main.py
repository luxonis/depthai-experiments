import argparse

import depthai as dai
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.host_fps_drawer import FPSDrawer
from host_node.normalize_bbox import NormalizeBbox
from video import Video

device = dai.Device()

modelDescription = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="r2-coco-512x288",
)
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-vid",
    "--vid",
    help="Provide Luxonis video name or path to video that should be used instead of camera",
    default="cars-tracking-above-01",
    type=str,
)
parser.add_argument(
    "-cam", "--cam", help="Use camera instead of video.", action="store_true"
)
args = parser.parse_args()


cam = args.cam
if not cam:
    video = args.vid
    video_path = Video(video).get_path()

with dai.Pipeline(device) as pipeline:
    nn_input_shape = (512, 288)

    if cam:
        cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        color_out = cam.requestOutput(
            size=nn_input_shape, type=dai.ImgFrame.Type.BGR888p
        )
    else:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(video_path)
        replay.setLoop(False)
        replay.setSize(nn_input_shape)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        img_manip = pipeline.create(dai.node.ImageManip)
        img_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        replay.out.link(img_manip.inputImage)
        color_out = img_manip.out

    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setNNArchive(nn_archive)
    color_out.link(nn.input)

    fps = pipeline.create(FPSDrawer).build(preview=color_out)
    normalize_bbox = pipeline.create(NormalizeBbox).build(frame=color_out, nn=nn.out)
    draw_detections = pipeline.create(DrawDetections).build(
        frame=fps.output, nn=normalize_bbox.output, label_map=nn.getClasses()
    )
    draw_detections.set_draw_labels(False)
    draw_detections.set_draw_confidence(False)
    draw_detections.set_color((0, 255, 0))
    display = pipeline.create(Display).build(frames=draw_detections.output)

    print("Pipeline created")
    pipeline.run()
    print("Pipeline finished")
