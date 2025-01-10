from pathlib import Path
import blobconverter
import depthai as dai
from host_age_gender import AgeGender
from detections_recognitions_sync import DetectionsRecognitionsSync
from host_display import Display


device = dai.Device()
# detection_model_description = dai.NNModelDescription(modelSlug="yunet", platform=device.getPlatform().name, modelVersionSlug="640x640")
recognition_model_description = dai.NNModelDescription(
    modelSlug="age-gender-recognition",
    platform=device.getPlatform().name,
    modelVersionSlug="62x62",
)
# detection_model_path = dai.getModelFromZoo(detection_model_description)
recognition_model_path = dai.getModelFromZoo(recognition_model_description)

face_det_model_description = dai.NNModelDescription(
    modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640"
)
face_det_archive_path = dai.getModelFromZoo(face_det_model_description)
face_det_nn_archive = dai.NNArchive(face_det_archive_path)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setPreviewNumFramesPool(10)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    face_manip = pipeline.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(300, 300)
    face_manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    cam.preview.link(face_manip.inputImage)

    face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    face_det_nn.setBlobPath(
        blobconverter.from_zoo(name="face-detection-retail-0004", shaves=5)
    )
    # face_det_nn.setNNArchive(face_det_nn_archive)
    face_det_nn.setBoundingBoxScaleFactor(0.8)
    face_det_nn.setDepthLowerThreshold(100)
    face_det_nn.setDepthUpperThreshold(5000)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.input.setMaxSize(1)
    face_manip.out.link(face_det_nn.input)
    stereo.depth.link(face_det_nn.inputDepth)

    script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(script.inputs["face_det_in"])
    cam.preview.link(script.inputs["preview"])
    script.setScriptPath(Path(__file__).parent / "script.py")

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(62, 62)
    recognition_manip.inputConfig.setWaitForMessage(True)
    script.outputs["manip_cfg"].link(recognition_manip.inputConfig)
    script.outputs["manip_img"].link(recognition_manip.inputImage)

    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setNNArchive(dai.NNArchive(recognition_model_path), 5)
    recognition_manip.out.link(recognition_nn.input)

    sync = pipeline.create(DetectionsRecognitionsSync).build()
    face_det_nn.out.link(sync.input_detections)
    recognition_nn.out.link(sync.input_recognitions)

    age_gender = pipeline.create(AgeGender).build(
        preview=cam.preview, detections_recognitions=sync.output
    )

    display = pipeline.create(Display).build(age_gender.output)
    display.setName("Preview")

    print("Pipeline created.")
    pipeline.run()
