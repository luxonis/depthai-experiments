import blobconverter
import depthai as dai

from blur_faces import BlurFaces


with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    # ColorCamera
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1080,1080)
    cam.setInterleaved(False)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6,
    ))
    # Link Face ImageManip -> Face detection NN node
    cam.preview.link(face_det_nn.input)

    objectTracker = pipeline.create(dai.node.ObjectTracker).build()
    objectTracker.setDetectionLabelsToTrack([1])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # Linking
    face_det_nn.passthrough.link(objectTracker.inputDetectionFrame)
    face_det_nn.passthrough.link(objectTracker.inputTrackerFrame)
    face_det_nn.out.link(objectTracker.inputDetections)
    # Send face detections to the host (for bounding boxes)
    
    pipeline.create(BlurFaces).build(
        video=cam.video,
        tracklets=objectTracker.out,
        tracker_passthrough=objectTracker.passthroughTrackerFrame
    )
    
    print("Pipeline created.")
    pipeline.run()
    print("Pipeline ended.")