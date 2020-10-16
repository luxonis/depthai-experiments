import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from time import sleep
import depthai
import cv2
from PIL import Image


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
server_HTTP = ThreadedHTTPServer(('localhost', 8090), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP.serve_forever)
th2.daemon = True
th2.start()

device = depthai.Device('', False)

pipeline = device.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": str(Path('./mobilenet-ssd/model.blob').resolve().absolute()),
        "blob_file_config": str(Path('./mobilenet-ssd/config.json').resolve().absolute())
    }
})

if pipeline is None:
    raise RuntimeError("Error initializing pipelne")

detections = []

while True:
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for detection in detections:
                left, top = int(detection.x_min * img_w), int(detection.y_min * img_h)
                right, bottom = int(detection.x_max * img_w), int(detection.y_max * img_h)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            server_TCP.datatosend = json.dumps([detection.get_dict() for detection in detections])
            server_HTTP.frametosend = frame
            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del pipeline
del device