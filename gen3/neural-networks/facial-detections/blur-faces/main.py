import blobconverter
import depthai as dai

from blur_faces import BlurFaces

face_det_model_description = dai.NNModelDescription(modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640")
face_det_archive_path = dai.getModelFromZoo(face_det_model_description)
face_det_nn_archive = dai.NNArchive(face_det_archive_path)

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    # ColorCamera
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1280, 800)
    cam.setInterleaved(False)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    #face_det_nn.setNNArchive(face_det_nn_archive) #TODO: swap in, when the HostNode->ObjectTracker is ready
    # Link Face ImageManip -> Face detection NN node
    cam.preview.link(face_det_nn.input)

    objectTracker = pipeline.create(dai.node.ObjectTracker)
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
        tracklets=objectTracker.out
    )
    
    print("Pipeline created.")
    pipeline.run()
    print("Pipeline ended.")
