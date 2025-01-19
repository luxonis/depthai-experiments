import depthai as dai
from host_stream_output import StreamOutput

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    vid_enc = pipeline.create(dai.node.VideoEncoder)
    vid_enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)
    cam.video.link(vid_enc.input)

    node = pipeline.create(StreamOutput).build(stream=vid_enc.bitstream)
    node.inputs["stream"].setBlocking(True)
    node.inputs["stream"].setMaxSize(30)

    print("Pipeline created. Watch the stream on rtsp://localhost:8554/preview")
    pipeline.run()
