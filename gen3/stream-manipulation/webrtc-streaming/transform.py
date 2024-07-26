import numpy as np
import depthai as dai
import blobconverter
import cv2

from aiortc import VideoStreamTrack
from av import VideoFrame

NN_SHAPES = {
    "face-detection-adas-0001" : (672, 384),
    "face-detection-retail-0004" : (300, 300),
    "mobilenet-ssd" : (300, 300),
    "pedestrian-and-vehicle-detector-adas-0001" : (672, 384),
    "pedestrian-detection-adas-0002" : (672, 384),
    "person-detection-retail-0013" : (544, 320),
    "person-vehicle-bike-detection-crossroad-1016" : (512, 512),
    "vehicle-detection-adas-0002" : (672, 384)
}

class VideoTransform(VideoStreamTrack):
    def __init__(self, pipeline, application, pc_id, options):
        super().__init__()
        self.application = application
        self.pc_id = pc_id

        self.nn_flag = options.nn
        self.depth_flag = (options.camera_type == "depth")

        self.pipeline = pipeline
        self.preview, self.nn, self.max_disparity = start_pipeline(self.pipeline, options)
        self.pipeline.start()

    async def recv(self):
        frame = await self.parse_frame()

        pts, time_base = await self.next_timestamp()
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame

    async def parse_frame(self):
        frame = self.preview.get().getFrame() if self.depth_flag else self.preview.get().getCvFrame()
        dets = self.nn.get().detections if self.nn is not None else []

        if self.depth_flag:
            frame = (frame * (255 / self.max_disparity)).astype(np.uint8)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        for detection in dets:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, f"LABEL {detection.label}", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, (255, 0, 0))
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        # Output the frame
        return frame

    def stop(self):
        print("Pipeline exited.")
        del self.pipeline
        super().stop()


def start_pipeline(pipeline, options):
    depth_flag = (options.camera_type == "depth")
    print("Creating pipeline...")

    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(options.width, options.height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(30)

    if options.nn:
        manip = pipeline.create(dai.node.ImageManip)
        manip.setKeepAspectRatio(False)
        manip.initialConfig.setResize(*NN_SHAPES[options.nn])
        cam.preview.link(manip.inputImage)

        nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(blobconverter.from_zoo(options.nn, shaves=6))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        manip.out.link(nn.input)

    if depth_flag:
        left = pipeline.create(dai.node.MonoCamera)
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        right = pipeline.create(dai.node.MonoCamera)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        if options.mono_camera_resolution == "THE_400_P":
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        elif options.mono_camera_resolution == "THE_720_P":
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        elif options.mono_camera_resolution == "THE_800_P":
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

        stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
        stereo.setConfidenceThreshold(200)
        stereo.setExtendedDisparity(options.extended_disparity)
        stereo.setSubpixel(options.subpixel)

        if options.median_filter == "MEDIAN_OFF":
            stereo.setSubpixelFractionalBits(5)
        else:
            stereo.setSubpixelFractionalBits(3)
        
        if options.median_filter == "MEDIAN_OFF":
            stereo.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        elif options.median_filter == "KERNEL_3x3":
            stereo.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
        elif options.median_filter == "KERNEL_5x5":
            stereo.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        elif options.median_filter == "KERNEL_7x7":
            stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    max_disparity = stereo.getMaxDisparity() if depth_flag else None
    # The host node architecture is not preferred here as the host node runs in a loop and
    #  we only want to get from the queues when pinged from the server in this experiment
    preview_q = (stereo.disparity if depth_flag else cam.preview).createOutputQueue(blocking=False, maxSize=4)
    nn_q = (nn.out.createOutputQueue(blocking=False, maxSize=4) if options.nn else None)

    print("Pipeline created.")
    return preview_q, nn_q, max_disparity


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
