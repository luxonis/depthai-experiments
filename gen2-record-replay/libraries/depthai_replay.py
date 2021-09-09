from pathlib import Path
import os
import cv2
import depthai as dai

class Replay:
    def __init__(self, path):
        self.path = path

        self.cap = {} # VideoCapture objects
        self.size = {} # Frame sizes
        self.lastFrame = {} # Last frame sent to the device

        # steam_types = ['color', 'left', 'right', 'depth']
        # extensions = ['mjpeg', 'avi', 'mp4']

        recordings = os.listdir(path)
        # Check if depth recording exists - if it does, don't create mono VideoCaptures
        if "left.mjpeg" in recordings and "right.mjpeg" in recordings:
            self.cap['left'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'left.mjpeg'))
            self.cap['right'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'right.mjpeg'))
        if "color.mjpeg" in recordings:
            self.cap['color'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'color.mjpeg'))

        if len(self.cap) == 0:
            raise RuntimeError("There are no .mjpeg recordings in the folder specified.")

        # Read basic info about the straems (resolution of streams etc.)
        for name in self.cap:
            self.size[name] = self.get_size(self.cap[name])

        self.color_size = None
        # By default crop image as needed to keep the aspect ratio
        self.keep_ar = True

    # Resize color frames prior to sending them to the device
    def set_resize_color(self, size):
        self.color_size = size
    def keep_aspect_ratio(self, keep_aspect_ratio):
        self.keep_ar = keep_aspect_ratio

    def resize_color(self, frame):
        if self.color_size is None:
            # No resizing needed
            return frame

        if not self.keep_ar:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, self.color_size)

        h = frame.shape[0]
        w = frame.shape[1]
        desired_ratio = self.color_size[0] / self.color_size[1]
        current_ratio = w / h

        # Crop width/heigth to match the aspect ratio needed by the NN
        if desired_ratio < current_ratio: # Crop width
            # Use full height, crop width
            new_w = (desired_ratio/current_ratio) * w
            crop = int((w - new_w) / 2)
            preview = frame[:, crop:w-crop]
        else: # Crop height
            # Use full width, crop height
            new_h = (current_ratio/desired_ratio) * h
            crop = int((h - new_h) / 2)
            preview = frame[crop:h-crop,:]

        return cv2.resize(preview, self.color_size)

    def init_pipeline(self):
        nodes = {}
        mono = 'left' and 'right' in self.cap
        depth = 'depth' in self.cap
        if mono and depth: # This should be possible either way.
            mono = False # Use depth stream by default

        pipeline = dai.Pipeline()

        nodes['color'] = pipeline.createXLinkIn()
        nodes['color'].setMaxDataSize(1920*1080*3)
        nodes['color'].setStreamName("color_in")

        if mono:
            nodes['left'] = pipeline.createXLinkIn()
            nodes['left'].setStreamName("left_in")
            nodes['right'] = pipeline.createXLinkIn()
            nodes['right'].setStreamName("right_in")

            nodes['stereo'] = pipeline.createStereoDepth()
            nodes['stereo'].initialConfig.setConfidenceThreshold(240)
            nodes['stereo'].setRectification(False)
            nodes['stereo'].setInputResolution(self.size['left'])

            nodes['left'].out.link(nodes['stereo'].left)
            nodes['right'].out.link(nodes['stereo'].right)

        if depth:
            nodes['depth'] = pipeline.createXLinkIn()
            nodes['depth'].setStreamName("depth_in")

        return pipeline, nodes

    def create_queues(self, device):
        self.queues = {}
        for name in self.cap:
            self.queues[name+'_in'] = device.getInputQueue(name+'_in')

    def to_planar(self, arr, shape = None):
        if shape is not None: arr = cv2.resize(arr, shape)
        return arr.transpose(2, 0, 1).flatten()

    def get_frames(self):
        frames = {}
        for name in self.cap:
            if not self.cap[name].isOpened(): return None
            ok, frame = self.cap[name].read()
            if ok:
                frames[name] = frame
        if len(frames) == 0: return None
        return frames

    def send_frames(self):
        frames = self.get_frames()
        if frames is None: return False # end of recording

        for name in frames:
            frame = frames[name]
            self.send_frame(frame, name)

        return True

    def get_size(self, cap):
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def send_frame(self, frame, name):
        q_name = name + '_in'
        if q_name in self.queues:
            if name == 'color':
                self.send_color(self.queues[q_name], frame)
            elif name == 'left':
                self.send_mono(self.queues[q_name], frame, False)
            elif name == 'right':
                self.send_mono(self.queues[q_name], frame, True)
            elif name == 'depth':
                self.send_depth(self.queues[q_name], frame)

    def send_mono(self, q, img, right):
        img = img[:,:,0] # all 3 planes are the same
        self.lastFrame['right' if right else 'left'] = img
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(img)
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if right else 1))
        q.send(frame)

    def send_color(self, q, img):
        # Resize/crop color frame as specified by the user
        img = self.resize_color(img)
        self.lastFrame['color'] = img
        h, w, c = img.shape
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(img))
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum(0)
        q.send(frame)

    def send_depth(self, q, depth):
        # TODO refactor saving depth. Reading will be from ROS bags.

        # print("depth size", type(depth))
        # depth_frame = np.array(depth).astype(np.uint8).view(np.uint16).reshape((400, 640))
        # depthFrameColor = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        # cv2.imshow("depth", depthFrameColor)
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        q.send(frame)