import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from socketserver import ThreadingMixIn
from time import sleep

import cv2
from PIL import Image

from depthai_sdk import OakCamera

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

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def callback(packet):
    frame = packet.frame
    visualizer = packet.visualizer

    for detection in packet.img_detections.detections:
        server_TCP.datatosend = str(detection.label)

    visualizer.draw(frame)
    server_HTTP.frametosend = frame


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', encode='mjpeg')
    nn = oak.create_nn('mobilenet-ssd', color)

    print(f"DepthAI SDK is up & running."
          f"Navigate to http://localhost:{HTTP_SERVER_PORT} with Chrome to see the mjpeg stream")

    oak.visualize(nn, callback=callback)
    oak.start(blocking=True)
