import blobconverter
import cv2
import depthai as dai

from palm_detection import PalmDetection


def crop_to_rect(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    delta = int((width - height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width - delta]


print("Creating pipeline...")
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(40)
cam.setIspScale(2, 3)  # 720P
cam.setVideoSize(720, 720)
cam.setPreviewSize(128, 128)
cam.setInterleaved(False)

isp_xout = pipeline.create(dai.node.XLinkOut)
isp_xout.setStreamName("cam")
cam.video.link(isp_xout.input)

print(f"Creating palm detection Neural Network...")
model_nn = pipeline.create(dai.node.NeuralNetwork)
model_nn.setBlobPath(blobconverter.from_zoo(name="palm_detection_128x128", zoo_type="depthai", shaves=6))
model_nn.input.setBlocking(False)
cam.preview.link(model_nn.input)

model_nn_xout = pipeline.create(dai.node.XLinkOut)
model_nn_xout.setStreamName("palm_nn")
model_nn.out.link(model_nn_xout.input)

print("Pipeline created.")

with dai.Device(pipeline) as device:
    # Create output queues
    vidQ = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    palmQ = device.getOutputQueue(name="palm_nn", maxSize=4, blocking=False)

    frame = None
    palmDetection = PalmDetection()

    while True:
        frame = vidQ.get().getCvFrame()
        palm_in = palmQ.tryGet()
        if palm_in is not None and frame is not None:
            try:
                palm_coords = palmDetection.decode(frame, palm_in)
                for bbox in palm_coords:
                    frame = cv2.rectangle(
                        img=frame,
                        pt1=(bbox[0], bbox[1]),
                        pt2=(bbox[2], bbox[3]),
                        color=(0, 127, 255),
                        thickness=4)
            except StopIteration:
                break
        cv2.imshow("Palm detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
