import depthai as dai
with dai.Device() as device:
    print(device.getMxId())