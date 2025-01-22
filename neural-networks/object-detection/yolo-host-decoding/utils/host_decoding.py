import depthai as dai
from .yolo_decode import decode_yolo_output


class HostDecoding(dai.node.HostNode):
    def __init__(self) -> None:
        self._conf_thresh = 0.3
        self._iou_thresh = 0.4
        self._nn_size = (512, 288)
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, nn: dai.NNData) -> "HostDecoding":
        self.link_args(nn)
        return self

    def set_conf_thresh(self, conf_thresh: float) -> None:
        self._conf_thresh = conf_thresh

    def set_iou_thresh(self, iou_thresh: float) -> None:
        self._iou_thresh = iou_thresh

    def set_nn_size(self, nn_size: tuple[int, int]) -> None:
        self._nn_size = nn_size

    def process(self, nn_data: dai.NNData) -> None:
        tensor_names = ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"]
        tensors = [
            nn_data.getTensor(
                tn, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW
            )
            for tn in tensor_names
        ]
        strides = [8, 16, 32]
        decoded = decode_yolo_output(tensors, strides, 0.5, 0.45, 80)
        dets = []
        for d in decoded:
            xmin, ymin, xmax, ymax, conf, cls = d
            det = dai.ImgDetection()
            det.label = int(cls)
            det.confidence = conf
            det.xmin = xmin / self._nn_size[0]
            det.ymin = ymin / self._nn_size[1]
            det.xmax = xmax / self._nn_size[0]
            det.ymax = ymax / self._nn_size[1]
            dets.append(det)
        img_dets = dai.ImgDetections()
        img_dets.detections = dets
        img_dets.setTimestamp(nn_data.getTimestamp())
        img_dets.setSequenceNum(nn_data.getSequenceNum())
        self.output.send(img_dets)
