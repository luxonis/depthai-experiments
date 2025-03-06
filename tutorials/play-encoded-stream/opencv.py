from pathlib import Path

import depthai as dai
from utils.arguments import initialize_argparser
from utils.decode_video_cv2 import DecodeVideoCV2
from utils.encoder_profiles import ENCODER_PROFILES

_, args = initialize_argparser()

if args.encode != "mjpeg":
    print("OpenCV decoding only supports MJPEG. Using MJPEG encoding.")
    args.encode = "mjpeg"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    platform = pipeline.getDefaultDevice().getPlatformAsString()

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        cam_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput((1920, 1080), type=dai.ImgFrame.Type.NV12)

    encoder = pipeline.create(dai.node.VideoEncoder).build(
        input=cam_out,
        frameRate=args.fps_limit,
        profile=ENCODER_PROFILES[args.encode],
    )

    decoder = pipeline.create(DecodeVideoCV2).build(enc_out=encoder.bitstream)

    visualizer.addTopic("Video", decoder.out)
    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
