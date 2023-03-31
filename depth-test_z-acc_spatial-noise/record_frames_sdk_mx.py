from depthai_sdk import OakCamera, RecordType
import depthai

FPS=10


def record_frames_sdk(path = './', fps=FPS, autoExposure=True, manualExposure=1000, iso=200, record=False):
    with OakCamera(args=False) as oak:
        left = oak.create_camera(source="cama,c" , resolution=depthai.ColorCameraProperties.SensorResolution.THE_1200_P, fps=fps)
        right = oak.create_camera(source="camb,c" , resolution=depthai.ColorCameraProperties.SensorResolution.THE_1200_P, fps=fps)
        vertical = oak.create_camera(source="camc,c" , resolution=depthai.ColorCameraProperties.SensorResolution.THE_1200_P, fps=fps)
        if autoExposure is False:
            left.node.initialControl.setManualExposure(manualExposure, iso)
            right.node.initialControl.setManualExposure(manualExposure, iso)
            vertical.node.initialControl.setManualExposure(manualExposure, iso)

        # Sync & save all streams
        if record:
            recorder = oak.record([left, right, vertical], path, RecordType.VIDEO)
        oak.visualize([left, right, vertical], scale=0.7)
        oak.start(blocking=True)


if __name__ == "__main__":
    record_frames_sdk()