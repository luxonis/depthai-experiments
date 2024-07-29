import depthai as dai
import threading
import cv2
import debugpy

def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def run_pipeline(pipeline : dai.Pipeline):
    pipeline.run()


class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()    


    def build(self, cam_out : dai.Node.Output, name : str) -> "Display":
        self.name = name
        self.link_args(cam_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, in_frame : dai.ImgFrame) -> None:

        cv2.imshow("rgb-" + self.name, in_frame.getCvFrame())
        if cv2.waitKey(1) == ord('q'):

            self.stopPipeline()


# This can be customized to pass multiple parameters
def getPipeline(dev : dai.Device, stereo : bool):
    pipeline = dai.Pipeline(dev)

    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # For the demo, just set a larger RGB preview size for OAK-D
    if stereo:
        cam_rgb.setPreviewSize(600, 300)
    else:
        cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    pipeline.create(Display).build(
        cam_out=cam_rgb.preview,
        name=dev.getMxId()
    )

    return pipeline


def pair_device_with_pipeline(dev_info, pipelines : list):
    device: dai.Device = dai.Device(dev_info)

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    pipelines.append(getPipeline(device, len(cameras)==3))


devices = filterInternalCameras(dai.Device.getAllAvailableDevices())
print(f'Found {len(devices)} internal devices')


pipelines = []
threads : list[threading.Thread] = []

for dev in devices:
    pair_device_with_pipeline(dev, pipelines)

for pipeline in pipelines:
    thread = threading.Thread(target=run_pipeline, args=(pipeline,))
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()

print('Devices closed')
