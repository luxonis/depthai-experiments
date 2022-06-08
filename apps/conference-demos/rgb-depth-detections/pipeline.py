import blobconverter
import depthai as dai

class HostSync:
    def __init__(self):
        self.dict = {}

    def add_msg(self, name, msg):
        seq = str(msg.getSequenceNum())
        if seq not in self.dict:
            self.dict[seq] = {}
        print(f"Adding {name} with seq `{seq}`")
        self.dict[seq][name] = msg

    def get_msgs(self):
        remove = []
        for name in self.dict:
            remove.append(name)
            if len(self.dict[name]) == 3:
                ret = self.dict[name]
                for rm in remove:
                    del self.dict[rm]
                return ret
        return None

class OakPipeline:
    def create_pipeline(self):
        def stereoPostProcessing(stereo):
            config = stereo.initialConfig.get()
            config.postProcessing.speckleFilter.enable = False
            config.postProcessing.speckleFilter.speckleRange = 50
            config.postProcessing.temporalFilter.enable = True
            config.postProcessing.spatialFilter.enable = True
            config.postProcessing.spatialFilter.holeFillingRadius = 2
            config.postProcessing.spatialFilter.numIterations = 1
            config.postProcessing.thresholdFilter.minRange = 350
            config.postProcessing.decimationFilter.decimationFactor = 1
            stereo.initialConfig.set(config)

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.initialControl.setManualFocus(130)
        camRgb.setIspScale(2, 3) # Downscale color to match mono
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        # camRgb.setPreviewKeepAspectRatio(False)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.video.link(xoutRgb.input)

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # stereoPostProcessing(stereo)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        nn = blobconverter.from_zoo(name="yolov4_tiny_coco_416x416", zoo_type="depthai", shaves=6)
        spatialDetectionNetwork.setBlobPath(nn)
        spatialDetectionNetwork.setConfidenceThreshold(0.3)
        # spatialDetectionNetwork.input.setBlocking(False)
        # spatialDetectionNetwork.input.setQueueSize(1)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(350)
        spatialDetectionNetwork.setDepthUpperThreshold(25000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(80)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
        spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
        spatialDetectionNetwork.setIouThreshold(0.5)

        camRgb.preview.link(spatialDetectionNetwork.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("detections")
        spatialDetectionNetwork.out.link(xoutNN.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
        return pipeline

    def start_thread(self, queue):
        with dai.Device(self.create_pipeline()) as device:
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb")
            detectionNNQueue = device.getOutputQueue(name="detections")
            depthQueue = device.getOutputQueue(name="depth")

            sync = HostSync()

            while True:
                if previewQueue.has():
                    sync.add_msg("rgb", previewQueue.get())
                if depthQueue.has():
                    sync.add_msg("depth", depthQueue.get())
                if detectionNNQueue.has():
                    sync.add_msg("detections", detectionNNQueue.get())

                msgs = sync.get_msgs()
                if msgs is not None:
                    queue.put(msgs)