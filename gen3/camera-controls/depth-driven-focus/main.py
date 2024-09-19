import depthai as dai
import blobconverter

from host_depth_driven_focus import DepthDrivenFocus

modelDescription = dai.NNModelDescription(modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640")
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1080, 1080)
    cam.setInterleaved(False)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.initialConfig.setConfidenceThreshold(240)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setExtendedDisparity(True)

    face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_nn.setConfidenceThreshold(0.4)
    #face_det_nn.setNNArchive(nn_archive) # TODO: swap in, when parsers can work with depth
    face_det_nn.setBoundingBoxScaleFactor(0.5)
    face_det_nn.setDepthLowerThreshold(200)
    face_det_nn.setDepthUpperThreshold(3000)

    cam.preview.link(face_det_nn.input)
    stereo.depth.link(face_det_nn.inputDepth)

    depth_driven_focus = pipeline.create(DepthDrivenFocus).build(
        preview=cam.video,
        control_queue=cam.inputControl.createInputQueue(),
        face_detection=face_det_nn.out,
        face_detection_depth=face_det_nn.passthroughDepth
    )

    print("Pipeline created.")
    pipeline.run()
