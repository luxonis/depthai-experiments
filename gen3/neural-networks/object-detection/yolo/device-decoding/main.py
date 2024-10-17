import depthai as dai

device = dai.Device()
visualizer = dai.RemoteConnection()
with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    nn = pipeline.create(dai.node.DetectionNetwork).build(
        cam,
        dai.NNModelDescription("yolov6-nano")
    )

    color_out = cam.requestOutput(
        size=(1280, 720), type=dai.ImgFrame.Type.BGR888i, fps=30
    )
    visualizer.addTopic("Color camera", color_out)
    visualizer.addTopic("Detections", nn.out)
    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")