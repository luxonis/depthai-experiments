import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
import sys
from socketserver import ThreadingMixIn
from time import sleep
import depthai as dai
import numpy as np
import cv2
from PIL import Image

HTTP_SERVER_PORT = 8090

class TCPServerRequest(socketserver.BaseRequestHandler):
    def handle(self):
        # Handle is called each time a client is connected
        # When OpenDataCam connects, do not return - instead keep the connection open and keep streaming data
        # First send HTTP header
        header = 'HTTP/1.0 200 OK\r\nServer: Mozarella/2.2\r\nAccept-Range: bytes\r\nConnection: close\r\nMax-Age: 0\r\nExpires: 0\r\nCache-Control: no-cache, private\r\nPragma: no-cache\r\nContent-Type: application/json\r\n\r\n'
        self.request.send(header.encode())
        while True:
            sleep(0.1)
            if hasattr(self.server, 'datatosend'):
                self.request.send(self.server.datatosend.encode() + "\r\n".encode())


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


# start TCP data server
server_TCP = socketserver.TCPServer(('localhost', 8070), TCPServerRequest)
th = threading.Thread(target=server_TCP.serve_forever)
th.daemon = True
th.start()


# start MJPEG HTTP Server
server_HTTP = ThreadedHTTPServer(('localhost', HTTP_SERVER_PORT), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP.serve_forever)
th2.daemon = True
th2.start()


# Get argument first
mobilenet_path = str((Path(__file__).parent / Path('mobilenet-ssd/model.blob')).resolve().absolute())
if len(sys.argv) > 1:
    mobilenet_path = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(mobilenet_path)
cam_rgb.preview.link(detection_nn.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# MobilenetSSD label texts
texts = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    print(f"DepthAI is up & running. Navigate to 'localhost:{str(HTTP_SERVER_PORT)}' with Chrome to see the mjpeg stream")

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    bboxes = []
    confidences = []
    labels = []

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
            shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
            frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if in_nn is not None:
            # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
            bboxes = np.array(in_nn.getFirstLayerFp16())
            # transform the 1D array into Nx7 matrix
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            # filter out the results which confidence less than a defined threshold
            bboxes = bboxes[bboxes[:, 2] > 0.5]
            # Cut bboxes and labels
            labels = bboxes[:, 1].astype(int)
            confidences = bboxes[:, 2]
            bboxes = bboxes[:, 3:7]

        if frame is not None:
            # if the frame is available, draw bounding boxes on it and show the frame
            for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                bbox = frame_norm(frame, raw_bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                server_TCP.datatosend = texts[label] +"," + f"{int(conf * 100)}%"
            server_HTTP.frametosend = frame

        if cv2.waitKey(1) == ord('q'):
            break