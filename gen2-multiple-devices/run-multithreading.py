import depthai as dai
import threading
import time
import cv2

alive = True

def getPipelineAndStreamList(id, mxid, usb_speed, cams):
    streams = []
    # TODO: optionally customize based on received parameters
    print(id, mxid, usb_speed.name, *[c.name for c in cams])
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(300, 300)
    camRgb.setInterleaved(True)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(3)

    name = f'rgb-{id}'
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName(name)
    streams.append(name)

    camRgb.preview.link(xoutRgb.input)

    return pipeline, streams

def worker(dev_info, id):
    with dai.Device(dai.OpenVINO.Version.VERSION_2021_4, dev_info) as dev:
        mxid = dev.getMxId()  # Note: for PoE is not the same as dev_info.getMxId()
        usbs = dev.getUsbSpeed()
        cams = dev.getConnectedCameras()
        pipeline, streams = getPipelineAndStreamList(id, mxid, usbs, cams)
        dev.startPipeline(pipeline)
        qlist = [dev.getOutputQueue(name=s, maxSize=4, blocking=False) for s in streams]

        global alive
        while alive:
            for q in qlist:
                pkt = q.get()
                tnow = dai.Clock.now().total_seconds()
                name = q.getName()
                tstamp = pkt.getTimestamp().total_seconds()
                seqnum = pkt.getSequenceNum()
                frame  = pkt.getCvFrame()
                latency_ms = (tnow - tstamp) * 1000
                print(f'{name:25}: seq:{seqnum:4} latency:{latency_ms:6.2f}ms')
                if 0:  # TODO fix multi-threading display
                    cv2.imshow(name, frame)
                    if cv2.waitKey(1) == ord('q'):
                        alive = False

device_infos = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')
threads = []
for dev in device_infos:
    id = dev.getMxId()  # Note: for PoE it's actually the IP address
    thread = threading.Thread(target=worker, args=(dev, id, ))
    thread.name += '-' + id
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()

print('Devices closed')
