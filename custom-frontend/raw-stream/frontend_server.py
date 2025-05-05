from http.server import HTTPServer
import os
import threading
from http.server import SimpleHTTPRequestHandler
from pathlib import Path


class FrontendServer:
    FILE = "index.html"

    def __init__(self, ip: str, port: int, directory: Path):
        self._ip = ip
        self._port = port
        self._directory = directory
        self._ensure_directory()
        self._httpd = HTTPServer((ip, port), self._get_handler())

    def _ensure_directory(self):
        if os.path.exists(self._directory / self.FILE):
            return
        raise FileNotFoundError(
            f"Failed to start the HTTP server. File {self.FILE} not found in the {self._directory} folder."
        )

    def _get_handler(self):
        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                super().__init__(*args, directory=self._directory, **kwargs)

        return CustomHandler

    def start(self):
        server_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        server_thread.start()

    def stop(self):
        self._httpd.shutdown()

    @property
    def ip(self):
        return self._ip

    @property
    def port(self):
        return self._port
