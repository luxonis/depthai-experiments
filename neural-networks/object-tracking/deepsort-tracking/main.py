from pathlib import Path

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.deepsort_tracking import DeepsortTracking
from utils.detection_crop_maker import DetectionCropMaker
from utils.detections_recognitions_sync import DetectionsRecognitionsSync

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(
        "luxonis/yolov6-nano:r2-coco-512x288", platform=device.getPlatform().name
    )
    detection_model_archive = dai.NNArchive(
        dai.getModelFromZoo(detection_model_description)
    )

    recognition_model_description = dai.NNModelDescription(
        "luxonis/osnet:imagenet-128x256", platform=device.getPlatform().name
    )
    recognition_model_archive = dai.NNArchive(
        dai.getModelFromZoo(recognition_model_description)
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
    if device.getPlatform().name == "RVC4":
        detection_resize.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
    cam_out.link(detection_resize.inputImage)

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        detection_resize.out, detection_model_archive
    )

    crop_maker = pipeline.create(DetectionCropMaker).build(
        detection_nn.out, detection_resize.out, recognition_model_archive.getInputSize()
    )
    crop_maker.set_confidence_threshold(0.5)
    recognition_manip = pipeline.create(dai.node.ImageManipV2)
    recognition_manip.initialConfig.setOutputSize(
        *recognition_model_archive.getInputSize()
    )
    recognition_manip.inputConfig.setWaitForMessage(True)
    crop_maker.out_cfg.link(recognition_manip.inputConfig)
    crop_maker.out_img.link(recognition_manip.inputImage)

    recognition_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        recognition_manip.out, recognition_model_archive
    )

    detection_recognitions_sync = pipeline.create(DetectionsRecognitionsSync).build()
    detection_nn.out.link(detection_recognitions_sync.input_detections)
    recognition_nn.out.link(detection_recognitions_sync.input_recognitions)

    deepsort_tracking = pipeline.create(DeepsortTracking).build(
        img_frames=detection_resize.out,
        detected_recognitions=detection_recognitions_sync.out,
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
