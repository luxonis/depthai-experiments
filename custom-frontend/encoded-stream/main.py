import depthai as dai


visualizer = dai.RemoteConnection(serveFrontend=False)
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    raw_stream = cam.requestOutput((640, 480), dai.ImgFrame.Type.NV12)
    visualizer.addTopic("Raw Stream", raw_stream)

    def custom_service(message):
        print("Received message:", message)
    visualizer.registerService("Custom Service", custom_service)
