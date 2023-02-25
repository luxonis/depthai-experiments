import blobconverter
import cv2
import depthai as dai
import numpy as np


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        self.arrays[name].append(msg)

    def get_msgs(self, seq):
        ret = {}
        for name, arr in self.arrays.items():
            for i, msg in enumerate(arr):
                if msg.getSequenceNum() == seq:
                    ret[name] = msg
                    self.arrays[name] = arr[i:]
                    break
        return ret


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    # ColorCamera
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1080, 1080)
    cam.setInterleaved(False)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("frame")
    cam.video.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6,
    ))
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

    pass_xout = pipeline.create(dai.node.XLinkOut)
    pass_xout.setStreamName("pass_out")
    objectTracker.passthroughTrackerFrame.link(pass_xout.input)

    tracklets_xout = pipeline.create(dai.node.XLinkOut)
    tracklets_xout.setStreamName("tracklets")
    objectTracker.out.link(tracklets_xout.input)
    print("Pipeline created.")
    return pipeline


with dai.Device(create_pipeline()) as device:
    frame_q = device.getOutputQueue("frame")
    tracklets_q = device.getOutputQueue("tracklets")
    pass_q = device.getOutputQueue("pass_out")
    sync = HostSync()
    while True:
        sync.add_msg("color", frame_q.get())

        # Using tracklets instead of ImgDetections in case NN inaccuratelly detected face, so blur
        # will still happen on all tracklets (even LOST ones)
        nn_in = tracklets_q.tryGet()
        if nn_in is not None:
            seq = pass_q.get().getSequenceNum()
            msgs = sync.get_msgs(seq)

            if not 'color' in msgs: continue
            frame = msgs["color"].getCvFrame()

            for t in nn_in.tracklets:
                # Expand the bounding box a bit so it fits the face nicely (also convering hair/chin/beard)
                t.roi.x -= t.roi.width / 10
                t.roi.width = t.roi.width * 1.2
                t.roi.y -= t.roi.height / 7
                t.roi.height = t.roi.height * 1.2

                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                bbox = [int(roi.topLeft().x), int(roi.topLeft().y), int(roi.bottomRight().x), int(roi.bottomRight().y)]

                face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                fh, fw, fc = face.shape
                frame_h, frame_w, frame_c = frame.shape

                # Create blur mask around the face
                mask = np.zeros((frame_h, frame_w), np.uint8)
                polygon = cv2.ellipse2Poly((bbox[0] + int(fw / 2), bbox[1] + int(fh / 2)), (int(fw / 2), int(fh / 2)),
                                           0, 0, 360, delta=1)
                cv2.fillConvexPoly(mask, polygon, 255)

                frame_copy = frame.copy()
                frame_copy = cv2.blur(frame_copy, (80, 80))
                face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                background_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=background_mask)
                # Blur the face
                frame = cv2.add(background, face_extracted)

            cv2.imshow("Frame", cv2.resize(frame, (900, 900)))

        if cv2.waitKey(1) == ord('q'):
            break
