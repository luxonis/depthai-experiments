from datetime import timedelta
import blobconverter
import cv2
import depthai as dai
import math

LENS_STEP = 3
DEBUG = False

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.5, self.bg_color, 6, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.5, self.color, 2, self.line_type)
    def rectangle(self, frame, x1,y1,x2,y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 6)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)

class HostSync:
    def __init__(self):
        self.arrays = {}
    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        self.arrays[name].append(msg)
    def get_msgs(self, timestamp):
        ret = {}
        for name, arr in self.arrays.items():
            for i, msg in enumerate(arr):
                time_diff = abs(msg.getTimestamp() - timestamp)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < timedelta(milliseconds=20):
                    ret[name] = msg
                    self.arrays[name] = arr[i:]
                    break
        return ret

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)

    # ColorCamera
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1080,1080)
    cam.setInterleaved(False)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')
    controlIn.out.link(cam.inputControl)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(240)
    stereo.setExtendedDisparity(True)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("frame")
    cam.video.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.4)
    face_det_nn.setBlobPath(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6,
        version='2021.3'
    ))

    face_det_nn.setBoundingBoxScaleFactor(0.5)
    face_det_nn.setDepthLowerThreshold(200)
    face_det_nn.setDepthUpperThreshold(3000)

    cam.preview.link(face_det_nn.input)
    stereo.depth.link(face_det_nn.inputDepth)

    pass_xout = pipeline.create(dai.node.XLinkOut)
    pass_xout.setStreamName("pass_out")
    face_det_nn.passthrough.link(pass_xout.input)

    nn_xout = pipeline.create(dai.node.XLinkOut)
    nn_xout.setStreamName("nn_out")
    face_det_nn.out.link(nn_xout.input)

    if DEBUG:
        bb_xout = pipeline.create(dai.node.XLinkOut)
        bb_xout.setStreamName('bb')
        face_det_nn.boundingBoxMapping.link(bb_xout.input)

        pass_xout = pipeline.create(dai.node.XLinkOut)
        pass_xout.setStreamName('pass')
        face_det_nn.passthroughDepth.link(pass_xout.input)
    print("Pipeline created.")
    return pipeline

def calculate_distance(coords):
    return math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)
def clamp(num, v0, v1):
    return max(v0, min(num, v1))
def get_lens_position(dist):
    # =150-A10*0.0242+0.00000412*A10^2
    return int(150 - dist * 0.0242 + 0.00000412 * dist**2)
def get_lens_position_lite(dist):
    # 141 + 0,0209x + −2E−05x^2
    return int(141 + dist * 0.0209 - 0.00002 * dist**2)

with dai.Device(create_pipeline()) as device:
    controlQ = device.getInputQueue('control')

    frame_q = device.getOutputQueue("frame", 4, False)
    nn_q = device.getOutputQueue("nn_out", 4, False)
    pass_q = device.getOutputQueue("pass_out", 4, False)
    if DEBUG:
        pass_q = device.getOutputQueue("pass", 4, False)
        bb_q = device.getOutputQueue("bb", 4, False)
    sync = HostSync()
    text = TextHelper()
    color = (220, 220, 220)

    lensPos = 150
    lensMin = 0
    lensMax = 255

    while True:
        sync.add_msg("color", frame_q.get())
        nn_in = nn_q.tryGet()
        if nn_in is not None:
            # Get NN output timestamp from the passthrough
            timestamp = pass_q.get().getTimestamp()
            msgs = sync.get_msgs(timestamp)

            if not 'color' in msgs: continue
            frame = msgs["color"].getCvFrame()

            height = frame.shape[0]
            width  = frame.shape[1]

            closest_dist = 99999999
            for detection in nn_in.detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)

                dist = int(calculate_distance(detection.spatialCoordinates))
                if dist < closest_dist: closest_dist = dist
                text.rectangle(frame, x1,y1,x2,y2)

            if closest_dist != 99999999:
                text.putText(frame,  "Face distance: {:.2f} m".format(closest_dist/1000), (330, 1045))
                new_lens_pos = clamp(get_lens_position_lite(closest_dist), lensMin, lensMax)
                if new_lens_pos != lensPos and new_lens_pos != 255:
                    lensPos = new_lens_pos
                    print("Setting manual focus, lens position: ", lensPos)
                    ctrl = dai.CameraControl()
                    ctrl.setManualFocus(lensPos)
                    controlQ.send(ctrl)
            else:
                text.putText(frame,  "Face distance: /", (330, 1045))
            text.putText(frame, f"Lens position: {lensPos}", (330, 1000))
            cv2.imshow("preview", cv2.resize(frame, (750,750)))

        if DEBUG:
            depth_in = pass_q.tryGet()
            if depth_in is not None:
                depthFrame = depth_in.getFrame()
                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                bb_in = bb_q.tryGet()
                if bb_in is not None:
                    roiDatas = bb_in.getConfigData()
                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
                        text.rectangle(depthFrameColor, xmin, ymin, xmax, ymax)
                cv2.imshow('depth', depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key in [ord(','), ord('.')]:
            if key == ord(','): lensPos -= LENS_STEP
            if key == ord('.'): lensPos += LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQ.send(ctrl)
