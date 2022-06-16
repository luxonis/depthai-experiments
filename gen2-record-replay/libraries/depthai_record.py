#!/usr/bin/env python3
from pathlib import Path
from multiprocessing import Queue
from threading import Thread
import depthai as dai
from enum import IntEnum

class EncodingQuality(IntEnum):
    BEST = 1 # Lossless MJPEG
    HIGH = 2 # MJPEG Quality=97 (default)
    MEDIUM = 3 # MJPEG Quality=93
    LOW = 4 # H265 BitrateKbps=10000

# class Recorder(IntEnum):
#     RAW = 1 # Save raw bitstream
#     MP4 = 2 # Containerize into mp4 file, requires `av` library

class Record():
    save = ['color', 'left', 'right']
    fps = 30
    timelapse = -1
    quality = EncodingQuality.HIGH
    rotate = -1
    preview = False

    def __init__(self, path: Path, device) -> None:
        self.device = device
        self.stereo = 1 < len(device.getConnectedCameras())
        self.mxid = device.getMxId()
        self.path = self.create_folder(path, self.mxid)

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.convert_mp4 = False

    def run(self):
        recorders = self.get_recorders()
        while True:
            try:
                frames = self.frame_q.get()
                if frames is None: # Terminate app
                    break
                for name in frames:
                    # Save all synced frames into files
                    recorders[name].write(name, frames[name])
            except KeyboardInterrupt:
                break
        # Close all recorders - Can't use ExitStack with VideoWriter
        for n in recorders:
            recorders[n].close()
        print('Exiting store frame thread')

    def get_recorders(self) -> dict:
        recorders = dict()
        save = self.save.copy()
        if 'depth' in save:
            from .video_recorders.rosbag_recorder import RosbagRecorder
            recorders['depth'] = RosbagRecorder(self.path, self.device, self.get_sizes())
            save.remove('depth')

        if len(save) == 0:
            print("only depth recorder")
            return recorders

        try:
            # Try importing av
            from .video_recorders.pyav_mp4_recorder import PyAvRecorder
            rec = PyAvRecorder(self.path, self.quality, self.fps)
        except:
            print("'av' library is not installed, depthai-record will save raw encoded streams.")
            from .video_recorders.raw_recorder import RawRecorder
            rec = RawRecorder(self.path, self.quality)

        # All other streams ("color", "left", "right", "disparity") will use
        # the same Raw/PyAv recorder
        for name in save:
            recorders[name] = rec
        return recorders

    def start(self):
        if not self.stereo: # If device doesn't have stereo camera pair
            if "left" in self.save: self.save.remove("left")
            if "right" in self.save: self.save.remove("right")
            if "disparity" in self.save: self.save.remove("disparity")
            if "depth" in self.save: self.save.remove("depth")

        if self.preview: self.save.append('preview')

        if 0 < self.timelapse:
            self.fps = 5

        self.pipeline, self.nodes = self.create_pipeline()

        self.frame_q = Queue(20)
        self.process = Thread(target=self.run)
        self.process.start()

        self.device.startPipeline(self.pipeline)

        self.queues = []
        maxSize = 1 if 0 < self.timelapse else 10
        for stream in self.save:
            self.queues.append({
                'q': self.device.getOutputQueue(name=stream, maxSize=maxSize, blocking=False),
                'msgs': [],
                'name': stream,
                'mxid': self.mxid
            })

    def set_fps(self, fps): self.fps = fps
    def set_timelapse(self, timelapse): self.timelapse = timelapse
    def set_quality(self, quality: EncodingQuality): self.quality = quality
    # def set_recorder(self, recorder: Recorder): self.recorder = recorder
    def set_preview(self, preview: bool): self.preview = preview
    # Which streams to save to the disk (on the host)
    def set_save_streams(self, save_streams): self.save = save_streams

    '''
    Available values for `angle`:
    - cv2.ROTATE_90_CLOCKWISE (0)
    - cv2.ROTATE_180 (1)
    - cv2.ROTATE_90_COUNTERCLOCKWISE (2)
    '''
    def set_rotate(self, angle):
        raise Exception("Rotating not yet supported!")
        # Currently RealSense Viewer throws error "memory access violation". Debug.
        self.rotate = angle

    def get_sizes(self):
        dict = {}
        if "color" in self.save: dict['color'] = self.nodes['color'].getVideoSize()
        if "right" in self.save: dict['right'] = self.nodes['right'].getResolutionSize()
        if "left" in self.save: dict['left'] = self.nodes['left'].getResolutionSize()
        if "disparity" in self.save: dict['disparity'] = self.nodes['left'].getResolutionSize()
        if "depth" in self.save: dict['depth'] = self.nodes['left'].getResolutionSize()
        return dict

    def create_folder(self, path: Path, mxid: str):
        i = 0
        while True:
            i += 1
            recordings_path = path / f"{i}-{str(mxid)}"
            if not recordings_path.is_dir():
                recordings_path.mkdir(parents=True, exist_ok=False)
                return recordings_path

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        nodes = {}

        def create_mono(name):
            nodes[name] = pipeline.create(dai.node.MonoCamera)
            nodes[name].setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
            nodes[name].setBoardSocket(socket)
            nodes[name].setFps(self.fps)

        def stream_out(name, fps, out, noEnc=False):
            # Create XLinkOutputs for the stream
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName(name)
            if noEnc:
                out.link(xout.input)
                return

            encoder = pipeline.create(dai.node.VideoEncoder)
            profile = dai.VideoEncoderProperties.Profile.H265_MAIN if self.quality == EncodingQuality.LOW else dai.VideoEncoderProperties.Profile.MJPEG
            encoder.setDefaultProfilePreset(fps, profile)

            if self.quality == EncodingQuality.BEST:
                encoder.setLossless(True)
            elif self.quality == EncodingQuality.HIGH:
                encoder.setQuality(97)
            elif self.quality == EncodingQuality.MEDIUM:
                encoder.setQuality(93)
            elif self.quality == EncodingQuality.LOW:
                encoder.setBitrateKbps(10000)

            out.link(encoder.input)
            encoder.bitstream.link(xout.input)

        if "color" in self.save:
            nodes['color'] = pipeline.create(dai.node.ColorCamera)
            nodes['color'].setBoardSocket(dai.CameraBoardSocket.RGB)
            # RealSense Viewer expects RGB color order
            nodes['color'].setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            nodes['color'].setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            nodes['color'].setIspScale(1,2) # 1080P
            nodes['color'].setFps(self.fps)

            if self.preview:
                nodes['color'].setPreviewSize(640, 360)
                stream_out("preview", None, nodes['color'].preview, noEnc=True)

            # TODO change out to .isp instead of .video when ImageManip will support I420 -> NV12
            # Don't encode color stream if we save depth; as we will be saving color frames in rosbags as well
            stream_out("color", nodes['color'].getFps(), nodes['color'].video) #, noEnc='depth' in self.save)

        if True in (el in ["left", "disparity", "depth"] for el in self.save):
            create_mono("left")
            if "left" in self.save:
                stream_out("left", nodes['left'].getFps(), nodes['left'].out)

        if True in (el in ["right", "disparity", "depth"] for el in self.save):
            create_mono("right")
            if "right" in self.save:
                stream_out("right", nodes['right'].getFps(), nodes['right'].out)

        if True in (el in ["disparity", "depth"] for el in self.save):
            nodes['stereo'] = pipeline.create(dai.node.StereoDepth)

            nodes['stereo'].initialConfig.setConfidenceThreshold(255)
            nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
            nodes['stereo'].setLeftRightCheck(True)
            nodes['stereo'].setExtendedDisparity(True)

            # if "disparity" not in self.save and "depth" in self.save:
            #     nodes['stereo'].setSubpixel(True) # For better depth visualization

            # if "depth" and "color" in self.save: # RGB depth alignment
            #     nodes['color'].setIspScale(1,3) # 4k -> 720P
            #     # For now, RGB needs fixed focus to properly align with depth.
            #     # This value was used during calibration
            #     nodes['color'].initialControl.setManualFocus(130)
            #     nodes['stereo'].setDepthAlign(dai.CameraBoardSocket.RGB)

            nodes['left'].out.link(nodes['stereo'].left)
            nodes['right'].out.link(nodes['stereo'].right)

            if "disparity" in self.save:
                stream_out("disparity", nodes['right'].getFps(), nodes['stereo'].disparity)
            if "depth" in self.save:
                stream_out('depth', None, nodes['stereo'].depth, noEnc=True)

        self.nodes = nodes
        self.pipeline = pipeline
        return pipeline, nodes

