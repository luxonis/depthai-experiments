import cv2
import depthai as dai
import numpy as np

from host_sync import HostSync


class BlurFaces(dai.node.HostNode):
    def __init__(self):
        self._sync = HostSync()
        super().__init__()


    def build(self, video: dai.Node.Output, tracklets: dai.Node.Output, tracker_passthrough: dai.Node.Output) -> "BlurFaces":
        self.link_args(video, tracklets, tracker_passthrough)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, frame: dai.ImgFrame, nn_in: dai.Tracklets, passthrough) -> None:
        self._sync.add_msg("color", frame)

        # Using tracklets instead of ImgDetections in case NN inaccuratelly detected face, so blur
        # will still happen on all tracklets (even LOST ones)
        if nn_in is not None:
            seq = passthrough.getSequenceNum()
            msgs = self._sync.get_msgs(seq)

            if not 'color' in msgs: return
            frame: np.ndarray = msgs["color"].getCvFrame()

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
                polygon = cv2.ellipse2Poly((bbox[0] + int(fw /2), bbox[1] + int(fh/2)), (int(fw /2), int(fh/2)), 0,0,360,delta=1)
                cv2.fillConvexPoly(mask, polygon, 255)

                frame_copy = frame.copy()
                frame_copy = cv2.blur(frame_copy, (80, 80))
                face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                background_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=background_mask)
                # Blur the face
                frame = cv2.add(background, face_extracted)

            cv2.imshow("Frame", cv2.resize(frame, (900,900)))

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()