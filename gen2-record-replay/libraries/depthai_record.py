#!/usr/bin/env python3
from pathlib import Path
from multiprocessing import Process, Queue
import depthai as dai
from enum import Enum

class EncodingQuality(Enum):
    BEST = 1 # Lossless MJPEG
    HIGH = 2 # MJPEG Quality=97 (default)
    MEDIUM = 3 # MJPEG Quality=93
    LOW = 4 # H265 BitrateKbps=10000

# path: Folder to where we save our streams
# frame_q: Queue of synced frames to be saved
# Nodes
# encode: Array of strings; which streams are encoded
def store_frames(path: str, frame_q, quality: EncodingQuality):
    files = {}

    def create_video_file(name):
        ext = 'h265' if quality == EncodingQuality.LOW else 'mjpeg'
        files[name] = open(str(path / f"{name}.{ext}"), 'wb')
        # if name == "color": fourcc = "I420"
        # elif name == "depth": fourcc = "Y16 " # 16-bit uncompressed greyscale image
        # else : fourcc = "GREY" #Simple, single Y plane for monochrome images.
        # files[name] = VideoWriter(str(path / f"{name}.avi"), VideoWriter_fourcc(*fourcc), fps, sizes[name], isColor=name=="color")

    while True:
        try:
            frames = frame_q.get()
            if frames is None:
                break
            for name in frames:

                if name not in files: # File wasn't created yet
                    create_video_file(name)

                files[name].write(frames[name])
                # frames[name].tofile(files[name])
        except KeyboardInterrupt:
            break
    # Close all files - Can't use ExitStack with VideoWriter
    for name in files:
        files[name].close()
    print('Exiting store frame process')

class Record:
    def __init__(self, path, device) -> None:
        self.save = ['color', 'left', 'right']
        self.fps = 30
        self.device = device
        self.quality = EncodingQuality.HIGH

        self.stereo = 1 < len(device.getConnectedCameras())
        self.path = self.create_folder(path, device.getMxId())

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.convert_mp4 = False

    def start_recording(self):
        if not self.stereo: # If device doesn't have stereo camera pair
            if "left" in self.save: self.save.remove("left")
            if "right" in self.save: self.save.remove("right")
            if "disparity" in self.save: self.save.remove("disparity")
            if "depth" in self.save: self.save.remove("depth")

        self.pipeline, self.nodes = self.create_pipeline()

        self.frame_q = Queue(20)
        self.process = Process(target=store_frames, args=(self.path, self.frame_q, self.quality))
        self.process.start()

        self.device.startPipeline(self.pipeline)

        self.queues = []
        for stream in self.save:
            self.queues.append({
                'q': self.device.getOutputQueue(name=stream, maxSize=10, blocking=False),
                'msgs': [],
                'name': stream
            })


    def set_fps(self, fps):
        self.fps = fps

    def set_quality(self, quality: EncodingQuality):
        self.quality = quality

    # Which streams to save to the disk (on the host)
    def set_save_streams(self, save_streams):
        self.save = save_streams
        print('save', self.save)

    def get_sizes(self):
        dict = {}
        if "color" in self.save: dict['color'] = self.nodes['color'].getVideoSize()
        if "right" in self.save: dict['right'] = self.nodes['right'].getResolutionSize()
        if "left" in self.save: dict['left'] = self.nodes['left'].getResolutionSize()
        if "disparity" in self.save: dict['disparity'] = self.nodes['left'].getResolutionSize()
        if "depth" in self.save: dict['depth'] = self.nodes['left'].getResolutionSize()
        return dict

    def create_folder(self, path, mxid):
        i = 0
        while True:
            i += 1
            recordings_path = Path(path) / f"{i}-{str(mxid)}"
            if not recordings_path.is_dir():
                recordings_path.mkdir(parents=True, exist_ok=False)
                return recordings_path

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        nodes = {}

        def create_mono(name):
            nodes[name] = pipeline.createMonoCamera()
            nodes[name].setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
            nodes[name].setBoardSocket(socket)
            nodes[name].setFps(self.fps)

        def stream_out(name, size, fps, out):
            # Create XLinkOutputs for the stream
            xout = pipeline.createXLinkOut()
            xout.setStreamName(name)

            encoder = pipeline.createVideoEncoder()
            profile = dai.VideoEncoderProperties.Profile.H265_MAIN if self.quality == EncodingQuality.LOW else dai.VideoEncoderProperties.Profile.MJPEG
            encoder.setDefaultProfilePreset(size, fps, profile)

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
            nodes['color'] = pipeline.createColorCamera()
            nodes['color'].setBoardSocket(dai.CameraBoardSocket.RGB)
            nodes['color'].setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            nodes['color'].setIspScale(1,2) # 1080P
            nodes['color'].setFps(self.fps)

            # TODO change out to .isp instead of .video when ImageManip will support I420 -> NV12
            stream_out("color", nodes['color'].getVideoSize(), nodes['color'].getFps(), nodes['color'].video)

        if "left" or "disparity" or "depth" in self.save:
            create_mono("left")
            if "left" in self.save:
                stream_out("left", nodes['left'].getResolutionSize(), nodes['left'].getFps(), nodes['left'].out)

        if "right" or "disparity" or "depth" in self.save:
            # Create mono cameras
            create_mono("right")
            if "right" in self.save:
                stream_out("right", nodes['right'].getResolutionSize(), nodes['right'].getFps(), nodes['right'].out)

        if "disparity" or "depth" in self.save:
            nodes['stereo'] = pipeline.createStereoDepth()
            nodes['stereo'].initialConfig.setConfidenceThreshold(255)
            nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
            nodes['stereo'].setLeftRightCheck(False)
            nodes['stereo'].setExtendedDisparity(False)
            nodes['stereo'].setSubpixel(False)

            nodes['left'].out.link(nodes['stereo'].left)
            nodes['right'].out.link(nodes['stereo'].right)

            if "disparity" in self.save:
                stream_out("disparity", nodes['right'].getResolutionSize(), nodes['right'].getFps(), nodes['stereo'].disparity)
            if "depth" in self.save:
                raise Exception("Depth recording isn't supported yet!")
                stream_out("depth", nodes['right'].getResolutionSize(), nodes['right'].getFps(), nodes['stereo'].depth)

        self.nodes = nodes
        self.pipeline = pipeline
        return pipeline, nodes

