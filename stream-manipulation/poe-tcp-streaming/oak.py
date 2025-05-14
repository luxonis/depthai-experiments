from pathlib import Path

import depthai as dai
from utils.oak_arguments import initialize_argparser
from utils.scripts import get_client_script, get_server_script

_, args = initialize_argparser()

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    platform = pipeline.getDefaultDevice().getPlatformAsString()

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(
            dai.ImgFrame.Type.BGR888i
            if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )
        replay.setLoop(True)
        replay.setFps(args.fps_limit)
        replay.setSize(1920, 1440)
        cam_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput((1920, 1440), fps=args.fps_limit)

    video_enc = pipeline.create(dai.node.VideoEncoder).build(
        cam_out,
        frameRate=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.MJPEG,
    )

    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)

    video_enc.bitstream.link(script.inputs["frame"])
    script.inputs["frame"].setBlocking(False)
    script.inputs["frame"].setMaxSize(1)

    if args.mode == "client":
        script.setScript(get_client_script(args.address))
    else:
        script.setScript(get_server_script())

    script.outputs["control"].link(cam.inputControl)

    print("Pipeline created.")
    pipeline.run()
