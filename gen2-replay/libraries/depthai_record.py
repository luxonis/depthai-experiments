#!/usr/bin/env python3
from pathlib import Path
from multiprocessing import Process, Queue
from cv2 import VideoWriter, VideoWriter_fourcc
import depthai as dai

class Record:
    def __init__(self, path, device) -> None:
        self.save = ['color', 'mono']
        self.fps = 30
        self.device = device

        self.stereo = 1 < len(device.getConnectedCameras())
        self.path = Path(self.create_folder(path, device.getMxId()))
        self.convert_mp4 = False

    def start_recording(self):
        if not self.stereo: # If device doesn't have stereo camera pair
            if "mono" in self.save:
                self.save.remove("mono")
            if "depth" in self.save:
                self.save.remove("depth")

        pipeline, nodes = self.create_pipeline()

        streams = []
        if "color" in self.save: streams.append("color")
        if "depth" in self.save: streams.append("depth")
        if "mono" in self.save:
            streams.append("left")
            streams.append("right")

        self.frame_q = Queue(20)
        self.process = Process(target=self.store_frames, args=())
        self.process.start()

        self.device.startPipeline(pipeline)

        self.queues = []
        for stream in streams:
            self.queues.append({
                'q': device.getOutputQueue(name=stream, maxSize=10, blocking=False),
                'msgs': [],
                'name': stream
            })


    def set_fps(self, fps):
        self.fps = fps

    def set_save_streams(self, save_streams):
        self.save = save_streams

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
        print(self.save)

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
            print('creating color xout')
            rgbOut = pipeline.createXLinkOut()
            rgbOut.setStreamName("color")
            rgb_encoder.bitstream.link(rgbOut.input)

        if "mono" or "depth" in self.save:
            # Create mono cameras
            nodes['left'] = pipeline.createMonoCamera()
            nodes['left'].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            nodes['left'].setBoardSocket(dai.CameraBoardSocket.LEFT)
            nodes['left'].setFps(self.fps)

            nodes['right'] = pipeline.createMonoCamera()
            nodes['right'].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            nodes['right'].setBoardSocket(dai.CameraBoardSocket.RIGHT)
            nodes['right'].setFps(self.fps)

            nodes['stereo'] = pipeline.createStereoDepth()
            nodes['stereo'].initialConfig.setConfidenceThreshold(240)
            nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
            nodes['stereo'].setLeftRightCheck(False)
            nodes['stereo'].setExtendedDisparity(False)
            nodes['stereo'].setSubpixel(False)
            # nodes['stereo'].setRectification(False)
            # nodes['stereo'].setRectifyMirrorFrame(False)

            nodes['left'].out.link(nodes['stereo'].left)
            nodes['right'].out.link(nodes['stereo'].right)

            if "depth" in self.save:
                depthOut = pipeline.createXLinkOut()
                depthOut.setStreamName("depth")
                nodes['stereo'].depth.link(depthOut.input)

            # Create output
            if "mono" in self.save:
                left_encoder = pipeline.createVideoEncoder()
                left_encoder.setDefaultProfilePreset(nodes['left'].getResolutionSize(), nodes['left'].getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
                # left_encoder.setLossless(True)
                nodes['stereo'].rectifiedLeft.link(left_encoder.input)
                # Create XLink output for left MJPEG stream
                leftOut = pipeline.createXLinkOut()
                leftOut.setStreamName("left")
                left_encoder.bitstream.link(leftOut.input)

                right_encoder = pipeline.createVideoEncoder()
                right_encoder.setDefaultProfilePreset(nodes['right'].getResolutionSize(), nodes['right'].getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
                # right_encoder.setLossless(True)
                nodes['stereo'].rectifiedRight.link(right_encoder.input)
                # Create XLink output for right MJPEG stream
                rightOut = pipeline.createXLinkOut()
                rightOut.setStreamName("right")
                right_encoder.bitstream.link(rightOut.input)

        self.nodes = nodes
        self.pipeline = pipeline
        return pipeline, nodes

    def store_frames(self):
        files = {}

        def create_video_file(name):
            fourcc = VideoWriter_fourcc(*'MJPG')
            width = self.nodes[name].getResolutionWidth()
            height = self.nodes[name].getResolutionHeight()
            path = str(self.path / f"{name}.mjpeg")
            # writer = VideoWriter(path, fourcc, self.fps, (width, height))
            # writer.release()
            # time.sleep(0.001)
            files[name] = open(path, 'wb')

        while True:
            try:
                frames = self.frame_q.get()
                if frames is None:
                    break
                for name in frames:

                    if name not in files: # File wasn't created yet
                        create_video_file(name)

                    files[name].write(frames[name])
                    # frames[name].tofile(files[name])
            except KeyboardInterrupt:
                break
        # Close all files
        for name in files:
            files[name].close()

        if self.convert_mp4:
            print("Converting .mjpeg to .mp4")
            for stream_name in files:
                self.convert_to_mp4((self.path / f"{stream_name}.mjpeg"))
        print('Exiting store frame process')

    def convert_to_mp4(self, convert):
        self.convert_mp4 = convert

    def mp4(path, deleteMjpeg = False):
        try:
            import ffmpy3
            path_output = path.parent / (path.stem + '.mp4')
            print(path_output)
            print(path)
            try:
                ff = ffmpy3.FFmpeg(
                    inputs={str(path): "-y"},
                    outputs={str(path_output): "-c copy -framerate {}".format(self.fps)}
                )
                print("Running conversion command... [{}]".format(ff.cmd))
                ff.run()
            except ffmpy3.FFExecutableNotFoundError:
                print("FFMPEG executable not found!")
            except ffmpy3.FFRuntimeError:
                print("FFMPEG runtime error!")
            print("Video conversion complete!")
        except ImportError:
            print("Module ffmpy3 not fouund!")
        except:
            print("Unknown error in convert_to_mp4!")
