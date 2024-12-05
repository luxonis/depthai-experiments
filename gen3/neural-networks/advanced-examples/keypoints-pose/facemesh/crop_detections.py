import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended

class CropDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._face_bbox_padding = 0.1


    def build(self, face_nn: dai.Node.Output, manipv2: bool) -> "CropDetections":
        if manipv2:
            self.output_cfg = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)])
        else:
            self.output_cfg = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)])
        self._manipv2 = manipv2
        self.link_args(face_nn)
        self.sendProcessingToPipeline(True)
        return self


    def set_face_bbox_padding(self, padding: float) -> None:
        self._face_bbox_padding = padding



    def process(self, face_nn: dai.Buffer) -> None:
        assert(isinstance(face_nn, ImgDetectionsExtended))
        face_dets = face_nn.detections
        if len(face_dets) == 0: return
        coords = face_dets[0] # take first
        
        coords.xmin -= self._face_bbox_padding
        coords.ymin -= self._face_bbox_padding
        coords.xmax += self._face_bbox_padding
        coords.ymax += self._face_bbox_padding
        
        self.limit_roi(coords)
        if self._manipv2:
            cfg = dai.ImageManipConfigV2()
            cfg.crop(coords.xmin, coords.ymin, (coords.xmin - coords.xmax), (coords.ymin - coords.ymax))
            cfg.resize(192, 192)
            cfg.setOutputSize(192, 192)
            self.output_cfg.send(cfg)
        else:
            cfg = dai.ImageManipConfig()
            cfg.setKeepAspectRatio(False)
            cfg.setCropRect(coords.xmin, coords.ymin, coords.xmax, coords.ymax)
            cfg.setResize(192, 192)
        self.output_cfg.send(cfg)


    def limit_roi(self, det: dai.ImgDetection) -> None:
        if det.xmin <= 0: det.xmin = 0.001
        if det.ymin <= 0: det.ymin = 0.001
        if det.xmax >= 1: det.xmax = 0.999
        if det.ymax >= 1: det.ymax = 0.999
