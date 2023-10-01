import depthai as dai
import cv2
import numpy as np

INPUT_NAME = "in"
OUTPUT_NAME = "out"

class DeviceEncoder:
    def create_pipeline(self, width: int, height: int, videoEncoderProfile: dai.VideoEncoderProperties.Profile, lossless: bool, bitrate_kbs: int, quality = 100) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        xIn = pipeline.create(dai.node.XLinkIn)
        videoEncoder = pipeline.create(dai.node.VideoEncoder)
        manip = pipeline.create(dai.node.ImageManip)
        xOut = pipeline.create(dai.node.XLinkOut)

        xIn.setStreamName(INPUT_NAME)
        xOut.setStreamName(OUTPUT_NAME)
        videoEncoder.setDefaultProfilePreset(30, videoEncoderProfile)
        videoEncoder.setLossless(lossless)
        videoEncoder.setBitrateKbps(bitrate_kbs)
        videoEncoder.setQuality(quality)
        manip.setMaxOutputFrameSize(int(width * height * 1.5))
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.NV12)

        # Linking
        xIn.out.link(manip.inputImage)
        manip.out.link(videoEncoder.input)
        videoEncoder.bitstream.link(xOut.input)
        return pipeline

    def __init__(self, width: int, height: int, videoEncoder: dai.VideoEncoderProperties.Profile, lossless: bool, bitrate_kbps: int, quality=100):
        self.device = dai.Device(self.create_pipeline(width, height, videoEncoder, lossless, bitrate_kbps, quality))
        self.inputQueue = self.device.getInputQueue(INPUT_NAME)
        self.outputQueue = self.device.getOutputQueue(OUTPUT_NAME, maxSize=30, blocking=True)

    def send_frame(self, frame: np.ndarray):
        img = dai.ImgFrame()
        img.setFrame(frame)
        img.setType(dai.RawImgFrame.Type.BGR888i)
        img.setWidth(frame.shape[1])
        img.setHeight(frame.shape[0])
        self.inputQueue.send(img)

    def get_frame(self):
        message: dai.ImgFrame = self.outputQueue.get()
        return message.getFrame()



if __name__ == "__main__":
    encoder = DeviceEncoder(640, 480, dai.VideoEncoderProperties.Profile.MJPEG, False)
    cap = cv2.VideoCapture(0)
    import av
    codec = av.CodecContext.create("mjpeg", "r")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        encoder.send_frame(frame)
        frame = encoder.get_frame()
        packets = codec.parse(frame)
        for packet in packets:
            for frame in codec.decode(packet):
                frame = frame.to_ndarray(format="bgr24")
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()