#!/usr/bin/env python3

import threading

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib


class RtspSystem(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(RtspSystem, self).__init__(**properties)
        self.data = None
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ! h265parse ! rtph265pay name=pay0 config-interval=1 name=pay0 pt=96'

    def send_data(self, data):
        self.data = data

    def start(self):
        t = threading.Thread(target=self._thread_rtsp)
        t.start()

    def _thread_rtsp(self):
        loop = GLib.MainLoop()
        loop.run()

    def on_need_data(self, src, length):
        if self.data is not None:
            retval = src.emit('push-buffer', Gst.Buffer.new_wrapped(self.data.tobytes()))
            if retval != Gst.FlowReturn.OK:
                print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class RTSPServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(RTSPServer, self).__init__(**properties)
        self.rtsp = RtspSystem()
        self.rtsp.set_shared(True)
        self.get_mount_points().add_factory("/preview", self.rtsp)
        self.attach(None)
        Gst.init(None)
        self.rtsp.start()

    def send_data(self, data):
        self.rtsp.send_data(data)


if __name__ == "__main__":
    import depthai as dai

    server = RTSPServer()

    pipeline = dai.Pipeline()

    FPS = 30
    colorCam = pipeline.create(dai.node.ColorCamera)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    colorCam.setFps(FPS)

    videnc = pipeline.create(dai.node.VideoEncoder)
    videnc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H265_MAIN)
    colorCam.video.link(videnc.input)

    veOut = pipeline.create(dai.node.XLinkOut)
    veOut.setStreamName("encoded")
    videnc.bitstream.link(veOut.input)

    device_infos = dai.Device.getAllAvailableDevices()
    if len(device_infos) == 0:
        raise RuntimeError("No DepthAI device found!")
    else:
        print("Available devices:")
        for i, info in enumerate(device_infos):
            print(f"[{i}] {info.getMxId()} [{info.state.name}]")
        if len(device_infos) == 1:
            device_info = device_infos[0]
        else:
            val = input("Which DepthAI Device you want to use: ")
            try:
                device_info = device_infos[int(val)]
            except:
                raise ValueError("Incorrect value supplied: {}".format(val))

    if device_info.protocol != dai.XLinkProtocol.X_LINK_USB_VSC:
        print("Running RTSP stream may be unstable due to connection... (protocol: {})".format(device_info.protocol))

    with dai.Device(pipeline, device_info) as device:
        encoded = device.getOutputQueue("encoded", maxSize=30, blocking=True)
        print("Setup finished, RTSP stream available under \"rtsp://localhost:8554/preview\"")
        while True:
            data = encoded.get().getData()
            server.send_data(data)
