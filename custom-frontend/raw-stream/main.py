import depthai as dai
from frontend_server import FrontendServer
from pathlib import Path


FRONTEND_DIRECTORY = Path(__file__).parent / "frontend" / "dist"
IP = "localhost"
PORT = 8080

frontend_server = FrontendServer(IP, PORT, FRONTEND_DIRECTORY)
print(f"Serving frontend at http://{IP}:{PORT}")
frontend_server.start()

visualizer = dai.RemoteConnection(serveFrontend=False)
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    raw_stream = cam.requestOutput((640, 480), dai.ImgFrame.Type.NV12)
    visualizer.addTopic("Raw Stream", raw_stream)

    def custom_service(message):
        print("Received message:", message)
    visualizer.registerService("Custom Service", custom_service)
    print("Running pipeline...")
    pipeline.run()