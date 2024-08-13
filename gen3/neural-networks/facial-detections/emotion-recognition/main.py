from pathlib import Path
import blobconverter
import depthai as dai
from detections_recognitions_sync import DetectionsRecognitionsSync
from emotions_recognition import DisplayEmotions


device = dai.Device()
recognition_model_description = dai.NNModelDescription(modelSlug="emotion-recognition", platform=device.getPlatform().name, modelVersionSlug="260x260")
recognition_model_path = dai.getModelFromZoo(recognition_model_description)

with dai.Pipeline(device) as pipeline:

    stereo = 1 < len(device.getConnectedCameras())

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1280, 800)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewNumFramesPool(15)
    cam.setFps(10)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
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

        # Spatial Detection network if OAK-D
        print("OAK-D detected, app will display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)
        stereo.depth.link(face_det_nn.inputDepth)
    else: # Detection network if OAK-1
        print("OAK-1 detected, app won't display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=5))
    face_det_manip.out.link(face_det_nn.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    # image_manip_script.setProcessor(dai.ProcessorType.LEON_CSS)
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])

    # Only send metadata, we are only interested in timestamp, so we can sync
    # depth frames with NN output
    # face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])
    cam.preview.link(image_manip_script.inputs['preview'])

    image_manip_script.setScriptPath(Path(__file__).parent / "script.py")

    manip_manip = pipeline.create(dai.node.ImageManip)
    manip_manip.initialConfig.setResize(260, 260) 
    manip_manip.inputConfig.setWaitForMessage(True)
    image_manip_script.outputs['manip_cfg'].link(manip_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(manip_manip.inputImage)

    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    emotions_nn = pipeline.create(dai.node.NeuralNetwork)
    emotions_nn.setNNArchive(dai.NNArchive(recognition_model_path), numShaves=5)
    emotions_nn.input.setBlocking(False)
    emotions_nn.input.setMaxSize(2)
    manip_manip.out.link(emotions_nn.input)

    sync_node = pipeline.create(DetectionsRecognitionsSync).build()
    sync_node.set_camera_fps(cam.getFps())

    face_det_nn.out.link(sync_node.input_detections)
    emotions_nn.out.link(sync_node.input_recognitions)

    pipeline.create(DisplayEmotions).build(
        rgb=cam.preview,
        detected_recognitions=sync_node.output,
        stereo=stereo
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
