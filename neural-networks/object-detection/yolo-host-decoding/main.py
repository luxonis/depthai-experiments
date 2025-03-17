import depthai as dai
from pathlib import Path

from utils.arguments import initialize_argparser
from utils.host_decoding import HostDecoding

_, args = initialize_argparser()

CAM_SIZE = (1280, 720)

conf_thresh = args.confidence_thresh
iou_thresh = args.iou_thresh

model_reference = "luxonis/yolov6-nano:r2-coco-512x288"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

platform = device.getPlatform().name
print(f"Platform: {platform}")

model_description = dai.NNModelDescription(model_reference)
model_description.platform = platform
nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

with dai.Pipeline(device) as pipeline:
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        replay.setFps(30)

    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            CAM_SIZE, dai.ImgFrame.Type.NV12, fps=args.fps_limit
        )

    input_node = replay.out if args.media_path else cam_out

    manip = pipeline.create(dai.node.ImageManipV2)
    manip.initialConfig.setOutputSize(*nn_archive.getInputSize())
    manip.initialConfig.setFrameType(frame_type)
    manip.setMaxOutputFrameSize(
        nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
    )
    input_node.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork).build(manip.out, nn_archive)

    host_decoding = pipeline.create(HostDecoding).build(nn=nn.out)
    host_decoding.set_nn_size(nn_archive.getInputSize())
    host_decoding.set_conf_thresh(conf_thresh)
    host_decoding.set_iou_thresh(iou_thresh)

    visualizer.addTopic("Camera", input_node)
    visualizer.addTopic("Detections", host_decoding.output)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
    print("Pipeline finished.")
