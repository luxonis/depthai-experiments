import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

import threading

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
