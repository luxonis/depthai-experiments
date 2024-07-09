import blobconverter
import depthai as dai

from kalman_filter_node import KalmanFilterNode


label_map = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


device = dai.Device()
print("Creating pipeline...")
with dai.Pipeline(device) as pipeline:
    cam_rgb = pipeline.create(dai.node.ColorCamera).build()
    detection_network = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork).build()
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth).build(mono_left.out, mono_right.out)
    object_tracker = pipeline.create(dai.node.ObjectTracker)

    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    detection_network.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=5))
    detection_network.setConfidenceThreshold(0.7)
    detection_network.input.setBlocking(False)
    detection_network.setBoundingBoxScaleFactor(0.5)
    detection_network.setDepthLowerThreshold(100)
    detection_network.setDepthUpperThreshold(5000)

    object_tracker.setDetectionLabelsToTrack([15])  # track only person
    object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    cam_rgb.preview.link(detection_network.input)
    q_rgb = object_tracker.passthroughTrackerFrame.createOutputQueue(4, False)
    q_tracklets = object_tracker.out.createOutputQueue(4, False)
    q_detections = object_tracker.passthroughDetections.createOutputQueue(4, False)
    
    detection_network.passthrough.link(object_tracker.inputTrackerFrame)
    detection_network.passthrough.link(object_tracker.inputDetectionFrame)
    detection_network.out.link(object_tracker.inputDetections)
    stereo.depth.link(detection_network.inputDepth)

    # Connect to device and start pipeline
    calibration_handler = device.readCalibration()
    baseline = calibration_handler.getBaselineDistance() * 10
    focal_length = calibration_handler.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 640, 400)[0][0]

    pipeline.create(KalmanFilterNode).build(
        rgb=object_tracker.passthroughTrackerFrame,
        tracker_out=object_tracker.out,
        baseline=baseline,
        focal_length=focal_length,
        label_map=label_map
    )

    print("Running pipeline...")
    pipeline.run()
    print("Pipeline stopped...")