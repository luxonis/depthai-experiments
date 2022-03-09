#!/usr/bin/env python3
import base64
import json
import math
import struct
import time
from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from threading import Thread

import depthai as dai
import numpy as np
from mcap.mcap0.writer import Writer


class EncodingQuality(Enum):
    BEST = 1  # Lossless MJPEG
    HIGH = 2  # MJPEG Quality=97 (default)
    MEDIUM = 3  # MJPEG Quality=93
    LOW = 4  # H265 BitrateKbps=10000


class Record:
    def __init__(self, path: Path, device, mcap) -> None:
        self.save = ['color', 'left', 'right']
        self.fps = 30
        self.timelapse = -1
        self.device = device
        self.quality = EncodingQuality.HIGH
        self.rotate = -1
        self.preview = False

        self.stereo = 1 < len(device.getConnectedCameras())
        self.mxid = device.getMxId()
        self.path = self.create_folder(path, self.mxid)

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.convert_mp4 = False
        self.mcap = mcap

    def run(self):
        files = {}
        mcaps = {}

        def create_video_file(name):
            if name == 'depth':  # or (name=='color' and 'depth' in self.save):
                files[name] = self.depthAiBag
            else:
                ext = 'h265' if self.quality == EncodingQuality.LOW else 'mjpeg'
                files[name] = open(str(self.path / f"{name}.{ext}"), 'wb')
            # if name == "color": fourcc = "I420"
            # elif name == "depth": fourcc = "Y16 " # 16-bit uncompressed greyscale image
            # else : fourcc = "GREY" #Simple, single Y plane for monochrome images.
            # files[name] = VideoWriter(str(path / f"{name}.avi"), VideoWriter_fourcc(*fourcc), fps, sizes[name],
            # isColor=name=="color")

        def create_mcap_file(name):
            mcaps[name] = Mcap(str(self.path / f"{name}"))
            mcaps[name].imageInit()

        while True:
            try:
                frames = self.frame_q.get()
                if frames is None:
                    break
                for name in frames:
                    if not self.mcap:
                        if name not in files:  # File wasn't created yet
                            create_video_file(name)
                        # if self.rotate != -1:  # Doesn't work atm
                        # frames[name] = cv2.rotate(frames[name], self.rotate)

                        files[name].write(frames[name])

                        # frames[name].tofile(files[name])

                    else:
                        if name not in mcaps:  # File wasn't created yet
                            create_mcap_file(name)

                        mcaps[name].imageSave(frames[name])

            except KeyboardInterrupt:
                break
        # Close all files - Can't use ExitStack with VideoWriter
        if not self.mcap:
            for name in files:
                files[name].close()
        else:
            for name in mcaps:
                mcaps[name].close()
        print('Exiting store frame thread')

    def start(self):
        if not self.stereo:  # If device doesn't have stereo camera pair
            if "left" in self.save: self.save.remove("left")
            if "right" in self.save: self.save.remove("right")
            if "disparity" in self.save: self.save.remove("disparity")
            if "depth" in self.save: self.save.remove("depth")

        if self.preview: self.save.append('preview')

        if 0 < self.timelapse:
            self.fps = 5

        self.pipeline, self.nodes = self.create_pipeline()

        if "depth" in self.save:
            from libraries.depthai_rosbags import DepthAiBags
            res = ['depth']
            # If rotate 90 degrees
            if self.rotate in [0, 2]: res = (res[1], res[0])
            self.depthAiBag = DepthAiBags(self.path, self.device, self.get_sizes(), rgb='color' in self.save)

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

    def set_fps(self, fps):
        self.fps = fps

    def set_timelapse(self, timelapse):
        self.timelapse = timelapse

    def set_quality(self, quality: EncodingQuality):
        self.quality = quality

    def set_preview(self, preview: bool):
        self.preview = preview

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
            nodes['color'].setIspScale(1, 2)  # 1080P
            nodes['color'].setFps(self.fps)

            if self.preview:
                nodes['color'].setPreviewSize(640, 360)
                stream_out("preview", None, nodes['color'].preview, noEnc=True)

            # TODO change out to .isp instead of .video when ImageManip will support I420 -> NV12
            # Don't encode color stream if we save depth; as we will be saving color frames in rosbags as well
            stream_out("color", nodes['color'].getFps(), nodes['color'].video)  # , noEnc='depth' in self.save)

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
            # TODO: configurable
            nodes['stereo'].setLeftRightCheck(True)
            nodes['stereo'].setExtendedDisparity(False)

            if "disparity" not in self.save and "depth" in self.save:
                nodes['stereo'].setSubpixel(True)  # For better depth visualization

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


class Mcap:

    # when initialising send in path (folder most likely: "./recordings/-name-") without .mcap at the end
    def __init__(self, path):
        self.fileName = path + ".mcap"
        self.stream = open(self.fileName, "wb")
        self.writer = Writer(self.stream)
        self.writer.start(profile="x-custom", library="my-writer-v1")
        self.channel_id = None

    def imageInit(self):
        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html
        schema_id = self.writer.register_schema(
            name="ros.sensor_msgs.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "object",
                            "properties": {
                                "stamp": {
                                    "type": "object",
                                    "properties": {
                                        "sec": {"type": "integer"},
                                        "nsec": {"type": "integer"},
                                    },
                                },
                            },
                        },
                        "format": {"type": "string"},
                        "data": {"type": "string", "contentEncoding": "base64"},
                    },
                },
            ).encode()
        )

        # create and register channel
        self.channel_id = self.writer.register_channel(
            schema_id=schema_id,
            topic="image",
            message_encoding="json",
        )

    def pointCloudInit(self):
        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html
        schema_id = self.writer.register_schema(
            "ros.sensor_msgs.PointCloud2",
            "jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "object",
                            "properties": {
                                "seq": {"type": "integer"},
                                "stamp": {
                                    "type": "object",
                                    "properties": {
                                        "sec": {"type": "integer"},
                                        "nsec": {"type": "integer"},
                                    },
                                },
                                "frame_id": {"type": "string"}
                            },
                        },
                        "height": {"type": "integer"},
                        "width": {"type": "integer"},
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "offset": {"type": "integer"},
                                    "datatype": {"type": "integer"},
                                    "count": {"type": "integer"}
                                }
                            },
                        },
                        "is_bigendian": {"type": "boolean"},
                        "point_step": {"type": "integer"},
                        "row_step": {"type": "integer"},
                        "data": {"type": "string", "contentEncoding": "base64"},
                        "is_dense": {"type": "boolean"}
                    },
                },
            ).encode("utf8")
        )

        # create and register channel
        self.channel_id = self.writer.register_channel(
            schema_id=schema_id,
            topic="pointClouds",
            message_encoding="json",
        )

    # send in image read with "cv2.getCvFrame"
    def imageSave(self, img):
        # convert cv2 image to .jpg format
        # is_success, im_buf_arr = cv2.imencode(".jpg", img)

        # read from .jpeg format to buffer of bytes
        # byte_im = im_buf_arr.tobytes()
        byte_im = img.tobytes()

        # data must be encoded in base64
        data = base64.b64encode(byte_im).decode("ascii")

        tmpTime = time.time_ns()
        sec = math.trunc(tmpTime / 1e9)
        nsec = tmpTime - sec

        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=tmpTime,
            data=json.dumps(
                {
                    "header": {"stamp": {"sec": sec, "nsec": nsec}},
                    "format": "jpeg",
                    "data": data,
                }
            ).encode("utf8"),
            publish_time=tmpTime,
        )

    # send in point cloud object read with
    # "o3d.io.read_point_cloud" or
    # "o3d.geometry.PointCloud.create_from_depth_image"
    # seq is just a sequence number that will be incremented in main  program (int from 0 to number at end of recording)
    def pointCloudSave(self, pcd, seq):
        points = np.asarray(pcd.points)

        # points must be read to a buffer and then encoded with base64
        buf = bytes()
        for point in points:
            buf += struct.pack('f', float(point[0]))
            buf += struct.pack('f', float(point[1]))
            buf += struct.pack('f', float(point[2]))

        data = base64.b64encode(buf).decode("ascii")

        tmpTime = time.time_ns()
        sec = math.trunc(tmpTime / 1e9)
        nsec = tmpTime - sec

        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=time.time_ns(),
            data=json.dumps(
                {
                    "header": {
                        "seq": seq,
                        "stamp": {"sec": sec, "nsec": nsec},
                        "frame_id": "front"
                    },
                    "height": 1,
                    "width": len(pcd),
                    "fields": [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                               {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                               {"name": "z", "offset": 8, "datatype": 7, "count": 1}],
                    "is_bigendian": False,
                    "point_step": 12,
                    "row_step": 12 * len(pcd),
                    "data": data,
                    "is_dense": True
                }
            ).encode("utf8"),
            publish_time=time.time_ns(),
        )

    def imuInit(self):
        # TODO create imu support
        return

    def imuSave(self):
        # TODO create imu support
        return

    def close(self):
        # end writer and close opened file
        self.writer.finish()
        self.stream.close()
