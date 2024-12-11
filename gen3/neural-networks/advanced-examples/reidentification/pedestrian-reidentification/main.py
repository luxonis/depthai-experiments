from pathlib import Path
import blobconverter
import depthai as dai

from fps_drawer import FPSDrawer
from detections_recognitions_sync import DetectionsRecognitionsSync
from pedestrian_reidentification import PedestrianReidentification

person_model_description = dai.NNModelDescription(modelSlug="scrfd-person-detection", platform="RVC2", modelVersionSlug="2-5g-640x640")
person_archive_path = dai.getModelFromZoo(person_model_description)
person_nn_archive = dai.NNArchive(person_archive_path)

recognition_model_description = dai.NNModelDescription(modelSlug="person-reidentification-retail", platform="RVC2", modelVersionSlug="0288")
recognition_archive_path = dai.getModelFromZoo(recognition_model_description)
recognition_nn_archive = dai.NNArchive(recognition_archive_path)

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1632, 960)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewNumFramesPool(10)
    cam.setFps(15)

    person_det_manip = pipeline.create(dai.node.ImageManip)
    person_det_manip.initialConfig.setResize(544, 320)
    person_det_manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    cam.preview.link(person_det_manip.inputImage)

    person_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    person_nn.setConfidenceThreshold(0.5)
    # person_nn.setBlobPath(blobconverter.from_zoo(name="person-detection-retail-0013", shaves=6))
    person_nn.setNNArchive(person_nn_archive)
    person_det_manip.out.link(person_nn.input)

    image_manip_script = pipeline.create(dai.node.Script)
    image_manip_script.setScriptPath(Path(__file__).parent / "script.py")
    cam.preview.link(image_manip_script.inputs['preview'])
    person_nn.out.link(image_manip_script.inputs['dets_in'])

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(128, 256)
    recognition_manip.inputConfig.setWaitForMessage(True)
    image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)

    print("Creating Recognition Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    # recognition_nn.setBlobPath(blobconverter.from_zoo(name="person-reidentification-retail-0288", shaves=6))
    recognition_nn.setNNArchive(recognition_nn_archive)
    recognition_manip.out.link(recognition_nn.input)

    detections_sync = pipeline.create(DetectionsRecognitionsSync).build()
    detections_sync.set_camera_fps(cam.getFps())
    person_nn.out.link(detections_sync.input_detections)
    recognition_nn.out.link(detections_sync.input_recognitions)

    fps_drawer = pipeline.create(FPSDrawer)
    cam.preview.link(fps_drawer.input)

    pedestrian_reidentification = pipeline.create(PedestrianReidentification).build(
        fps_drawer.output, 
        detections_sync.output
    )
    
    pipeline.run()