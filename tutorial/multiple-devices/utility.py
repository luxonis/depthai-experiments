import depthai as dai

def filter_internal_cameras(devices : list[dai.DeviceInfo]) -> list[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def run_pipeline(pipeline : dai.Pipeline) -> None:
    pipeline.run()