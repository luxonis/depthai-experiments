#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np


class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, rgb_out : dai.Node.Output, det_out : dai.Node.Output) -> "Display":
        self.link_args(rgb_out, det_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, in_frame : dai.Buffer, nn_data : dai.Buffer) -> None:
        assert(isinstance(in_frame, dai.ImgFrame))
        assert(isinstance(nn_data, dai.ImgDetections))

        frame = in_frame.getCvFrame()
        dets = nn_data.detections

        for detection in dets:
            bbox = self.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def frame_norm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


with dai.Pipeline() as pipeline:

    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(1280, 800) # High res frames for visualization on host
    camRgb.setInterleaved(False)
    camRgb.setFps(40)

    downscale_manip = pipeline.create(dai.node.ImageManip)
    downscale_manip.initialConfig.setResize(300, 300)
    camRgb.preview.link(downscale_manip.inputImage)

    # Define a neural network that will make predictions based on the source frames
    detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    detection_nn.setConfidenceThreshold(0.5)
    detection_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    # Frames will come a lot faster than NN will be able to process. To always
    # run inference on latest frame, we can set its queue size to 1.
    # Default is Blocking and queue size 5.
    detection_nn.input.setBlocking(False)
    detection_nn.input.setMaxSize(1)
    downscale_manip.out.link(detection_nn.input) # Downscaled 300x300 frames


    pipeline.create(Display).build(
        rgb_out=camRgb.preview,
        det_out=detection_nn.out
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")