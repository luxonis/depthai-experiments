# This Python file uses the following encoding: utf-8
import asyncio
import threading
import time
from pathlib import Path

import aiohttp
import cv2
import depthai as dai
from aiohttp import web, MultipartWriter
from depthai_sdk import PipelineManager

host = "localhost"
port = 9001

class HttpHandler:
    static_path = Path(__file__).parent / "build"
    site = None
    loop = None
    datatosend = None
    app = None

    def __init__(self, instance, loop):
        self.instance = instance
        self.loop = loop
        self.app = web.Application(middlewares=[self.static_serve])
        self.app.add_routes([
            web.get('/stream', self.stream),
            web.post('/update', self.update),
        ])

    @web.middleware
    async def static_serve(self, request, handler):
        relative_file_path = Path(request.path).relative_to('/')  # remove root '/'
        file_path = self.static_path / relative_file_path  # rebase into static dir
        if not file_path.exists():
            return await handler(request)
        if file_path.is_dir():
            file_path /= 'index.html'
            if not file_path.exists():
                return web.FileResponse(Path(__file__).parent / "404.html")
        return web.FileResponse(file_path)

    def run(self):
        self.runner = web.AppRunner(self.app)
        self.loop.run_until_complete(self.runner.setup())
        self.site = aiohttp.web.TCPSite(self.runner, host, port)
        self.loop.run_until_complete(self.site.start())
        self.loop.run_forever()

    def close(self):
        self.loop.run_until_complete(self.runner.cleanup())

    async def update(self, request):
        data = await request.json()
        if data.get('capture', '') == "true":
            self.instance.pm.captureStill()
        if data.get('autofocus', '') == "trigger":
            self.instance.pm.triggerAutoFocus()
        if 'autofocus' in data:
            if hasattr(dai.CameraControl.AutoFocusMode, data['autofocus']):
                self.instance.pm.updateColorCamConfig(autofocus=getattr(dai.CameraControl.AutoFocusMode, data['autofocus']))
            else:
                return web.HTTPBadRequest(reason=f"Parameter 'autofocus' contains non existing value \"{data['autofocus']}\". Check dai.CameraControl.AutoFocusMode for available values")
        if data.get('autoexposure', '') == "true":
            self.instance.pm.triggerAutoExposure()
        if data.get('autowhitebalance', '') == "true":
            print("Auto white-balance enable")
            self.instance.pm.triggerAutoWhiteBalance()

        self.instance.pm.updateColorCamConfig(
            focus=data['focus'] if 'focus' in data else None,
            exposure=data['expiso'][0] if 'expiso' in data else None,
            sensitivity=data['expiso'][1] if 'expiso' in data else None,
            whitebalance=data['whitebalance'] if 'whitebalance' in data else None,
            saturation=data['saturation'] if 'saturation' in data else None,
            brightness=data['brightness'] if 'brightness' in data else None,
            contrast=data['contrast'] if 'contrast' in data else None,
            sharpness=data['sharpness'] if 'sharpness' in data else None,
        )
        return web.Response()

    async def stream(self, request):
        boundary = 'boundarydonotcross'
        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=--{}'.format(boundary),
        })
        try:
            await response.prepare(request)
            while True:
                if self.datatosend is not None:
                    with MultipartWriter('image/jpeg', boundary=boundary) as mpwriter:
                        mpwriter.append(str(self.datatosend), {
                            'Content-Type': 'image/jpeg'
                        })
                        await mpwriter.write(response, close_boundary=False)
                    await response.drain()
        except ConnectionResetError:
            print("Client connection closed")
        finally:
            return response


class WebApp:
    def __init__(self):
        super().__init__()
        self.running = False
        self.webserver = None
        self.selectedPreview = "color"
        self.thread = None
        self.pm = None

    def shouldRun(self):
        return self.running

    def updatePreview(self, selected):
        self.selectedPreview = selected

    def runDemo(self):
        self.pm = PipelineManager(lowBandwidth=True)
        self.pm.createColorCam(xout=True, xoutStill=True)

        with dai.Device(self.pm.pipeline) as device:
            previewQueue = device.getOutputQueue(self.pm.nodes.xoutRgb.getStreamName())
            stillQueue = device.getOutputQueue(self.pm.nodes.xoutRgbStill.getStreamName())
            savedPath = Path(__file__).parent / "saved"

            while self.shouldRun():
                previewPacket = previewQueue.tryGet()
                if previewPacket is not None:
                    self.webserver.datatosend = previewPacket.getData()
                stillPacket = stillQueue.tryGet()
                if stillPacket is not None:
                    savedPath.mkdir(exist_ok=True)
                    cv2.imwrite(str(savedPath / f"saved-{time.monotonic()}.jpg"), stillPacket.getCvFrame())
                time.sleep(1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.runDemo)
        self.thread.daemon = True
        self.thread.start()

        if self.webserver is None:
            loop = asyncio.get_event_loop()
            self.webserver = HttpHandler(self, loop)
            print("Server started http://{}:{}".format(host, port))

            try:
                self.webserver.run()
            except KeyboardInterrupt:
                pass

            self.webserver.close()
        else:
            self.webserver.frametosend = None

    def stop(self):
        self.running = False
        self.thread.join()

    def restartDemo(self):
        self.stop()
        self.start()


if __name__ == "__main__":
    repo_root = Path(__file__).parent.resolve().absolute()
    import subprocess
    subprocess.check_call(["yarn"], cwd=repo_root)
    subprocess.check_call(["yarn", "build"], cwd=repo_root)
    WebApp().start()
