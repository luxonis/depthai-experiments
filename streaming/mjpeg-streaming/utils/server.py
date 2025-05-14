from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from time import sleep

import cv2


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-type", "multipart/x-mixed-replace; boundary=--jpgboundary"
        )
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, "frametosend"):
                ok, encoded = cv2.imencode(".jpg", self.server.frametosend)
                self.wfile.write("--jpgboundary".encode())
                self.send_header("Content-type", "image/jpeg")
                self.send_header("Content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    pass
