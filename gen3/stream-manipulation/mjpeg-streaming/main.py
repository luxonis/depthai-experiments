import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from time import sleep
import depthai as dai
import cv2
import blobconverter

from host_stream_output import StreamOutput

HTTP_SERVER_PORT = 8090

# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                ok, encoded = cv2.imencode('.jpg', self.server.frametosend)
                self.wfile.write("--jpgboundary".encode())
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


# start MJPEG HTTP Server
server = ThreadedHTTPServer(('localhost', HTTP_SERVER_PORT), VideoStreamHandler)
th = threading.Thread(target=server.serve_forever)
th.daemon = True
th.start()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork).build()
    nn.setBlobPath(blobconverter.from_zoo("mobilenet-ssd", shaves=5))
    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)
    nn.setBoundingBoxScaleFactor(0.5)
    nn.setDepthLowerThreshold(100)
    nn.setDepthUpperThreshold(5000)
    cam.preview.link(nn.input)
    stereo.depth.link(nn.inputDepth)

    output = pipeline.create(StreamOutput).build(
        preview=cam.preview,
        depth=nn.passthroughDepth,
        nn=nn.out,
        server=server
    )
    output.inputs["preview"].setBlocking(False)
    output.inputs["preview"].setMaxSize(4)
    output.inputs["depth"].setBlocking(False)
    output.inputs["depth"].setMaxSize(4)
    output.inputs["nn"].setBlocking(False)
    output.inputs["nn"].setMaxSize(4)

    print(f"Pipeline created. Navigate to 'localhost:{str(HTTP_SERVER_PORT)}' with Chrome to see the mjpeg stream")
    pipeline.run()
