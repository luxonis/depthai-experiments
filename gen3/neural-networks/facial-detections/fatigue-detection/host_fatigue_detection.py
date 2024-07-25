import cv2
import depthai as dai

from face_landmarks import FaceLandmarks

class FatigueDetection(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.decode = FaceLandmarks()

    def build(self, preview: dai.Node.Output, face_nn: dai.Node.Output, landmarks_nn: dai.Node.Output) -> "FatigueDetection":
        self.link_args(preview, face_nn, landmarks_nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, face_dets: dai.ImgDetections, landmarks: dai.NNData) -> None:
        frame = preview.getCvFrame()

        dets = face_dets.detections
        if len(dets) > 0:
            detection = dets[0]
            face_coords = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            self.decode.run_land68(frame, landmarks, face_coords)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
