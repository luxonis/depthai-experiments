import depthai as dai
from detections_recognitions_sync import DetectionsRecognitionsSync
from emotions_recognition import DisplayEmotions
from depthai_nodes import YuNetParser
from host_node.depth_merger import DepthMerger
from host_node.crop_detections import CropDetections

device = dai.Device()
device_platform = device.getPlatform()

modelSlug = "gray-64x64" if device.getPlatform() == dai.Platform.RVC2 else "260x260"
recognition_model_description = dai.NNModelDescription(model=f"luxonis/emotion-recognition:{modelSlug}", platform=device_platform.name)
recognition_model_path = dai.getModelFromZoo(recognition_model_description)

with dai.Pipeline(device) as pipeline:
    stereo = 1 < len(device.getConnectedCameras())
    
    output_dimensions = (640, 640)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    if device.getPlatform() == dai.Platform.RVC2:
        model_dimensions = (320, 320)
        nn_archive = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription(modelSlug="yunet", platform=device_platform.name, modelVersionSlug=f"{model_dimensions[0]}x{model_dimensions[1]}")))
        face_det_nn = pipeline.create(dai.node.NeuralNetwork)
        face_det_nn.setNNArchive(nn_archive, numShaves=4)
        cam.requestOutput(model_dimensions, dai.ImgFrame.Type.BGR888p).link(face_det_nn.input)
    else:
        model_dimensions = (640, 640)
        face_det_nn = pipeline.create(dai.node.NeuralNetwork).build(
            cam,
            dai.NNModelDescription(model=f"yunet:{model_dimensions[0]}x{model_dimensions[1]}")
        )

    cam_out = cam.requestOutput((output_dimensions), dai.ImgFrame.Type.NV12)
    parser = pipeline.create(YuNetParser)
    parser.setConfidenceThreshold(0.5)
    face_det_nn.out.link(parser.input)

    if stereo:
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B).requestOutput(output_dimensions, dai.ImgFrame.Type.NV12)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C).requestOutput(output_dimensions, dai.ImgFrame.Type.NV12)
        
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left,
            right,
            presetMode=dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        if device_platform == dai.Platform.RVC2:
            stereo.setOutputSize(output_dimensions[0], output_dimensions[1])

        # Spatial Detection network if OAK-D
        print("OAK-D detected, app will display spatial coordiantes")
        depth_merger = pipeline.create(DepthMerger).build(parser.out, stereo.depth, device.readCalibration())
        depth_merger.host_spatials_calc.setLowerThreshold(100)
        depth_merger.host_spatials_calc.setUpperThreshold(5000)
        nn_out = depth_merger.output
    else: # Detection network if OAK-1
        print("OAK-1 detected, app won't display spatial coordiantes")
        nn_out = parser.out

    nn_input_type = dai.ImgFrame.Type.GRAY8 if device_platform == dai.Platform.RVC2 else dai.ImgFrame.Type.BGR888i
    nn_input_shape = (64, 64) if device_platform == dai.Platform.RVC2 else (260, 260)
    crop_detections = pipeline.create(CropDetections).build(nn_out)
    crop_detections.set_resize(nn_input_shape)
    crop_detections.set_frame_type(nn_input_type)

    detection_manip = pipeline.create(dai.node.ImageManipV2)
    detection_manip.initialConfig.setFrameType(nn_input_type)
    detection_manip.initialConfig.addResize(*nn_input_shape)
    detection_manip.inputConfig.setWaitForMessage(False)
    crop_detections.output_config.link(detection_manip.inputConfig)
    cam_out.link(detection_manip.inputImage)

    emotions_nn = pipeline.create(dai.node.NeuralNetwork)
    emotions_nn.setNNArchive(dai.NNArchive(recognition_model_path))
    emotions_nn.input.setBlocking(False)
    emotions_nn.input.setMaxSize(2)
    detection_manip.out.link(emotions_nn.input)

    sync_node = pipeline.create(DetectionsRecognitionsSync)

    nn_out.link(sync_node.input_detections)
    emotions_nn.out.link(sync_node.input_recognitions)

    pipeline.create(DisplayEmotions).build(
        rgb=cam_out,
        detected_recognitions=sync_node.output,
        stereo=stereo
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
