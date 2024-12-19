import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended

class ProcessDetections(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        
        self.config_output = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            img_detections = self.detections_input.get()
            detections = img_detections.detections
            # w, h = img_detections.transformation.getSize()
            w, h = 1728, 960
            
            configs_message = dai.MessageGroup()
            for i, detection in enumerate(detections):
                cfg = dai.ImageManipConfigV2()
                detection: dai.ImgDetection = detection
                rect = dai.RotatedRect()
                rect.center.x = (detection.xmin + detection.xmax) / 2
                rect.center.y = (detection.ymin + detection.ymax) / 2
                rect.size.width = detection.xmax - detection.xmin
                rect.size.height = detection.ymax - detection.ymin
                rect.angle = 0
                cfg.addCropRotatedRect(rect=rect, normalizedCoords=True)
                cfg.addResize(256, 256)
                # rect = detection.rotated_rect
                # rect = rect.denormalize(w, h)
                # cfg.addCropRotatedRect(rect, normalizedCoords=False)
                # cfg.setOutputSize(256, 256)
                # cfg.setReusePreviousImage(False)
                cfg.setTimestamp(img_detections.getTimestamp())
                
                configs_message[str(i+100)] = cfg
                            
            configs_message.setTimestamp(img_detections.getTimestamp())
            self.config_output.send(configs_message)