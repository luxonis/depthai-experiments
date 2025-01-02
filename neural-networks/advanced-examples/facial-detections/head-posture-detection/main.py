from pathlib import Path
import depthai as dai

from head_posture_detection import HeadPostureDetection
from detections_recognitions_sync import DetectionsRecognitionsSync

face_det_model_description = dai.NNModelDescription(modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640")
face_det_archive_path = dai.getModelFromZoo(face_det_model_description)
face_det_nn_archive = dai.NNArchive(face_det_archive_path)

recognition_model_description = dai.NNModelDescription(modelSlug="head-pose-estimation-adas", platform="RVC2", modelVersionSlug="0001")
recognition_archive_path = dai.getModelFromZoo(recognition_model_description)
recognition_nn_archive = dai.NNArchive(recognition_archive_path)

device = dai.Device()
stereo = 1 < len(device.getConnectedCameras())
with dai.Pipeline(device) as pipeline:
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewNumFramesPool(10)

    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    cam.preview.link(face_det_manip.inputImage)

    if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        print("OAK-D detected, app will display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)
        stereo.depth.link(face_det_nn.inputDepth)
    else:
        print("OAK-1 detected, app won't display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setNNArchive(face_det_nn_archive)
    face_det_nn.input.setBlocking(False)
    face_det_nn.input.setMaxSize(1)
    face_det_manip.out.link(face_det_nn.input)

    image_manip_script = pipeline.create(dai.node.Script)
    image_manip_script.setScriptPath(Path(__file__).parent / 'script.py')
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])
    cam.preview.link(image_manip_script.inputs['preview'])

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(60, 60)
    recognition_manip.inputConfig.setWaitForMessage(True)
    image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)

    print("Creating recognition Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setNNArchive(recognition_nn_archive)
    recognition_manip.out.link(recognition_nn.input)

    recognition_sync = pipeline.create(DetectionsRecognitionsSync).build()
    face_det_nn.out.link(recognition_sync.input_detections)
    recognition_nn.out.link(recognition_sync.input_recognitions)
    recognition_sync.set_camera_fps(cam.getFps())
    
    head_pose_detection = pipeline.create(HeadPostureDetection).build(cam.preview, recognition_sync.output)
    head_pose_detection.inputs["detected_recognitions"].setBlocking(False)
    head_pose_detection.inputs["detected_recognitions"].setMaxSize(1)

    pipeline.run()
