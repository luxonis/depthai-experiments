#!/usr/bin/env python3
from pathlib import Path
from multiprocessing import Process, Queue
from cv2 import VideoWriter, VideoWriter_fourcc
import types
import os
import depthai as dai
import contextlib

# path: Folder to where we save our streams
# frame_q: Queue of synced frames to be saved
# Nodes
# encode: Array of strings; which streams are encoded
def store_frames(path: str, frame_q, encode, fps, sizes):
    files = {}

    def create_video_file(name):
        if name in encode:
            files[name] = open(str(path / f"{name}.mjpeg"), 'wb')
        else:
            files[name] = VideoWriter(str(path / f"{name}.avi"), VideoWriter_fourcc(*'DIVX'), fps, sizes[name])
        # fourcc = VideoWriter_fourcc(*'MJPG')
        # writer = VideoWriter(path, fourcc, self.fps, (width, height))
        # writer.release()
        # time.sleep(0.001)

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
        if isinstance(files[name], VideoWriter):
            print('release videowriter')
            files[name].release()
        else: files[name].close()
    print('Exiting store frame process')

class Record:
    def __init__(self, path, device) -> None:
        self.save = ['color', 'mono']
        self.fps = 30
        self.device = device

        self.stereo = 1 < len(device.getConnectedCameras())
        self.path = self.create_folder(path, device.getMxId())

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.convert_mp4 = False

    def start_recording(self):
        if not self.stereo: # If device doesn't have stereo camera pair
            if "mono" in self.save:
                self.save.remove("mono")
            if "disparity" in self.save:
                self.save.remove("disparity")

        self.pipeline, self.nodes = self.create_pipeline()

        streams = []
        if "color" in self.save: streams.append("color")
        if "disparity" in self.save: streams.append("disparity")
        if "mono" in self.save:
            streams.append("left")
            streams.append("right")

        self.frame_q = Queue(20)
        self.process = Process(target=store_frames, args=(self.path, self.frame_q, self.encode, self.fps, self.get_sizes()))
        self.process.start()

        self.device.startPipeline(self.pipeline)

        self.queues = []
        for stream in streams:
            self.queues.append({
                'q': self.device.getOutputQueue(name=stream, maxSize=10, blocking=False),
                'msgs': [],
                'name': stream
            })


    def set_fps(self, fps):
        self.fps = fps

    # Which streams to save to the disk (on the host)
    def set_save_streams(self, save_streams):
        self.save = save_streams

    # Which streams to encode
    def set_encoding(self, encode_streams):
        self.encode = encode_streams

    def get_sizes(self):
        dict = {}
        if "color" in self.save:
            dict['color'] = self.nodes['color'].getVideoSize()
        if "mono" in self.save:
            dict['left'] = self.nodes['left'].getResolutionSize()
            dict['right'] = self.nodes['right'].getResolutionSize()
        if "disparity" in self.save:
            dict['disparity'] = self.nodes['left'].getResolutionSize()
        print(dict)
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

        if "color" in self.save:
            nodes['color'] = pipeline.createColorCamera()
            nodes['color'].setBoardSocket(dai.CameraBoardSocket.RGB)
            nodes['color'].setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            nodes['color'].setFps(self.fps)

            rgb_encoder = pipeline.createVideoEncoder()
            rgb_encoder.setDefaultProfilePreset(nodes['color'].getVideoSize(), nodes['color'].getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            # rgb_encoder.setLossless(True)
            nodes['color'].video.link(rgb_encoder.input)

            # Create output for the rgb
            rgbOut = pipeline.createXLinkOut()
            rgbOut.setStreamName("color")
            rgb_encoder.bitstream.link(rgbOut.input)

        if "mono" or "disparity" in self.save:
            # Create mono cameras
            nodes['left'] = pipeline.createMonoCamera()
            nodes['left'].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            nodes['left'].setBoardSocket(dai.CameraBoardSocket.LEFT)
            nodes['left'].setFps(self.fps)

            nodes['right'] = pipeline.createMonoCamera()
            nodes['right'].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            nodes['right'].setBoardSocket(dai.CameraBoardSocket.RIGHT)
            nodes['right'].setFps(self.fps)

            if "disparity" in self.save:
                nodes['stereo'] = pipeline.createStereoDepth()
                nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
                nodes['stereo'].setLeftRightCheck(False)
                nodes['stereo'].setExtendedDisparity(False)
                nodes['stereo'].setSubpixel(False)

                nodes['left'].out.link(nodes['stereo'].left)
                nodes['right'].out.link(nodes['stereo'].right)

                disparityOut = pipeline.createXLinkOut()
                disparityOut.setStreamName("disparity")

                if "disparity" in self.encode:
                    disp_encoder = pipeline.createVideoEncoder()
                    disp_encoder.setDefaultProfilePreset(nodes['left'].getResolutionSize(), nodes['left'].getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
                    nodes['stereo'].disparity.link(disp_encoder.input)
                    disp_encoder.bitstream.link(disparityOut.input)
                else:
                    nodes['stereo'].disparity.link(disparityOut.input)

            if "mono" in self.save:
                # Create XLinkOutputs for mono streams
                leftOut = pipeline.createXLinkOut()
                leftOut.setStreamName("left")
                rightOut = pipeline.createXLinkOut()
                rightOut.setStreamName("right")
                if "mono" in self.encode:
                    # Encode mono streams
                    left_encoder = pipeline.createVideoEncoder()
                    left_encoder.setDefaultProfilePreset(nodes['left'].getResolutionSize(), nodes['left'].getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
                    # left_encoder.setLossless(True)
                    # left_encoder.setQuality(50)
                    nodes['left'].out.link(left_encoder.input)

                    right_encoder = pipeline.createVideoEncoder()
                    right_encoder.setDefaultProfilePreset(nodes['right'].getResolutionSize(), nodes['right'].getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
                    # right_encoder.setLossless(True)
                    # right_encoder.setQuality(50)
                    nodes['right'].out.link(right_encoder.input)

                    left_encoder.bitstream.link(leftOut.input)
                    right_encoder.bitstream.link(rightOut.input)
                else:
                    # Don't encode
                    nodes['left'].out.link(leftOut.input)
                    nodes['right'].out.link(rightOut.input)

        self.nodes = nodes
        self.pipeline = pipeline
        return pipeline, nodes

