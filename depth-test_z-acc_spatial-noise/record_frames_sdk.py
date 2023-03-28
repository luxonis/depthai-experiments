from depthai_sdk import OakCamera, RecordType
import depthai

FPS=10


def record_frames_sdk(path = './'):
    with OakCamera() as oak:
        # color = oak.create_camera('color', resolution='1080P', fps=30, encode='MJPEG')
        # color.config_color_camera(isp_scale=(2, 3)) # 720P
        left = oak.create_camera(source="camb,c" , resolution=depthai.ColorCameraProperties.SensorResolution.THE_1200_P, fps=FPS)
        right = oak.create_camera(source="camc,c" , resolution=depthai.ColorCameraProperties.SensorResolution.THE_1200_P, fps=FPS)
        vertical = oak.create_camera(source="camd,c" , resolution=depthai.ColorCameraProperties.SensorResolution.THE_1200_P, fps=FPS)

        # Sync & save all streams
        recorder = oak.record([left, right, vertical], path, RecordType.VIDEO)
        oak.visualize([left, right, vertical])
        oak.start(blocking=True)


if __name__ == "__main__":
    record_frames_sdk()