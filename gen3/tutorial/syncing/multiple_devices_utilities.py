import threading
import depthai as dai
import cv2


def filter_internal_cameras(devices : list[dai.DeviceInfo]) -> list[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


class OpencvManager:
    def __init__(self) -> None:
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.keys = []


    def run(self) -> None:
        while True:
            self.newFrameEvent.wait()
            for name in self.frames.keys():
                if self.frames[name] is not None:
                    cv2.imshow(name, self.frames[name])

                    if cv2.waitKey(1) == ord('q'):
                        return
    

    def set_frame(self, frame : dai.ImgFrame, name : int) -> None:
        with self.lock:
            self.frames[name] = frame
            self.newFrameEvent.set()


    def set_custom_keys(self, keys : list[int]) -> None:
        self.keys.extend(keys)
        self.frames = self._init_frames()

    
    def _init_frames(self) -> dict:
        dic = dict()
        for key in self.keys:
            dic[key] = None
        return dic