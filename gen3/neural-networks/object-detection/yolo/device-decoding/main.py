import depthai as dai
import cv2

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

    color_out = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.BGR888i, fps=30) # Would be better to use NV12, but visualizer doesn't support strided NV12 yet
    visualizer.addTopic("Camera", color_out)
    visualizer.addTopic("Detections", nn.out)
    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
    print("Pipeline finished.")
