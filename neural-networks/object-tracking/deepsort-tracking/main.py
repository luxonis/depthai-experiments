from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
from depthai_nodes.node.utils import generate_script_content
from utils.arguments import initialize_argparser
from utils.deepsort_tracking import DeepsortTracking

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

if args.fps_limit is None:
    args.fps_limit = 4 if platform == "RVC2" else 30
    print(
        f"FPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag."
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(
        "luxonis/yolov6-nano:r2-coco-512x288", platform=platform
    )
    detection_model_archive = dai.NNArchive(
        dai.getModelFromZoo(detection_model_description)
    )

    embeddings_model_description = dai.NNModelDescription(
        "luxonis/osnet:imagenet-128x256", platform=platform
    )
    embeddings_model_archive = dai.NNArchive(
        dai.getModelFromZoo(embeddings_model_description)
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        cam_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            size=(1920, 1080),
            type=dai.ImgFrame.Type.NV12,
            fps=args.fps_limit,
        )
    detection_resize = pipeline.create(dai.node.ImageManipV2)
    detection_resize.setMaxOutputFrameSize(
        detection_model_archive.getInputWidth()
        * detection_model_archive.getInputHeight()
        * 3
    )
    detection_resize.initialConfig.setOutputSize(
        detection_model_archive.getInputWidth(),
        detection_model_archive.getInputHeight(),
        dai.ImageManipConfigV2.ResizeMode.STRETCH,
    )
    detection_resize.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    if platform == "RVC4":
        detection_resize.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
    cam_out.link(detection_resize.inputImage)

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        detection_resize.out, detection_model_archive
    )

    script = pipeline.create(dai.node.Script)
    detection_nn.out.link(script.inputs["det_in"])
    detection_nn.passthrough.link(script.inputs["preview"])
    script_content = generate_script_content(
        resize_width=embeddings_model_archive.getInputWidth(),
        resize_height=embeddings_model_archive.getInputHeight(),
        padding=0,
    )
    script.setScript(script_content)

    embeddings_manip = pipeline.create(dai.node.ImageManipV2)
    embeddings_manip.initialConfig.setOutputSize(
        embeddings_model_archive.getInputWidth(),
        embeddings_model_archive.getInputHeight(),
    )

    script.outputs["manip_cfg"].link(embeddings_manip.inputConfig)
    script.outputs["manip_img"].link(embeddings_manip.inputImage)

    embeddings_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        embeddings_manip.out, embeddings_model_archive
    )

    gather_data = pipeline.create(GatherData).build(camera_fps=args.fps_limit)
    detection_nn.out.link(gather_data.input_reference)
    embeddings_nn.out.link(gather_data.input_data)

    deepsort_tracking = pipeline.create(DeepsortTracking).build(
        img_frame=detection_resize.out,
        gathered_data=gather_data.out,
        labels=detection_model_archive.getConfigV1().model.heads[0].metadata.classes,
    )

    visualizer.addTopic("Video", cam_out, "images")
    visualizer.addTopic("Detections", deepsort_tracking.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
