import blobconverter
import cv2
import depthai as dai
import numpy as np

DECODE = True


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type)

    def rectangle(self, frame, p1, p2):
        cv2.rectangle(frame, p1, p2, self.bg_color, 6)
        cv2.rectangle(frame, p1, p2, self.color, 1)


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setPreviewSize(1080, 1080)
camRgb.setInterleaved(False)
camRgb.initialControl.setManualFocus(145)
camRgb.setFps(60)

frameOut = pipeline.create(dai.node.XLinkOut)
frameOut.setStreamName("color")
camRgb.preview.link(frameOut.input)

# 1080x1080 -> 384x384 required by the model
scale_manip = pipeline.create(dai.node.ImageManip)
scale_manip.initialConfig.setResize(384, 384)
scale_manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
camRgb.preview.link(scale_manip.inputImage)

nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.3)
nn.setBlobPath(blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai", shaves=6))
nn.input.setQueueSize(1)
nn.input.setBlocking(False)
scale_manip.out.link(nn.input)

# Linking
nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to a device and start the pipeline
with dai.Device(pipeline) as device:
    qRight = device.getOutputQueue("color", maxSize=4, blocking=False)
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)
    c = TextHelper()


    def decode(frame, bbox, detector=None):
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        data, vertices_array, binary_qrcode = detector.detectAndDecode(img)
        if data:
            print("Decoded text", data)
            return data
        else:
            print("Decoding failed")
            return ""


    def expandDetection(det, percent=2):
        percent /= 100
        det.xmin -= percent
        det.ymin -= percent
        det.xmax += percent
        det.ymax += percent
        if det.xmin < 0: det.xmin = 0
        if det.ymin < 0: det.ymin = 0
        if det.xmax > 1: det.xmax = 1
        if det.ymax > 1: det.ymax = 1


    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    if DECODE: detector = cv2.QRCodeDetector()

    while True:
        frame = qRight.get().getCvFrame()
        detections = inDet = qDet.get().detections

        for det in detections:
            expandDetection(det)
            bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
            c.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]))
            c.putText(frame, f"{int(det.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 20))
            if DECODE:
                text = decode(frame, bbox, detector)
                c.putText(frame, text, (bbox[0] + 10, bbox[1] + 40))

        cv2.imshow("Image", frame)

        if cv2.waitKey(1) == ord('q'):
            break
