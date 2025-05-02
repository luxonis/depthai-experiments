from typing import Tuple

import depthai as dai


class StereoConfigController(dai.node.HostNode):
    def __init__(self):
        super().__init__()

        self._current_cfg = dai.StereoDepthConfig()
        self._current_cfg.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self._current_cfg.setLeftRightCheck(True)
        self._current_cfg.setConfidenceThreshold(15)
        self._current_cfg.setSubpixelFractionalBits(3)

        self.out_cfg = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.StereoDepthConfig, True)
            ]
        )
        self.out_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

    def build(self, preview: dai.Node.Output):
        self.link_args(preview)

        return self

    def _create_text_annot(self, text: str, pos: Tuple[float, float]):
        txt_annot = dai.TextAnnotation()
        txt_annot.fontSize = 15
        txt_annot.backgroundColor = dai.Color(0, 1, 0, 1)
        txt_annot.textColor = dai.Color(0, 0, 0, 1)
        txt_annot.position = dai.Point2f(*pos)
        txt_annot.text = text
        return txt_annot

    def process(self, frame: dai.ImgFrame):
        img_annots = dai.ImgAnnotations()
        img_annot = dai.ImgAnnotation()

        conf_thresh = self._create_text_annot(
            f'Confidence threshold: {self._current_cfg.getConfidenceThreshold()} (keys: ",", ".")',
            (0.05, 0.05),
        )
        median_filter = self._create_text_annot(
            f'Median filter: {self._current_cfg.getMedianFilter().name} (key: "k")',
            (0.05, 0.09),
        )
        lr_check = self._create_text_annot(
            f'Left-right check: {self._current_cfg.getLeftRightCheck()} (key: "l")',
            (0.05, 0.13),
        )

        img_annot.texts.append(conf_thresh)
        img_annot.texts.append(median_filter)
        img_annot.texts.append(lr_check)
        img_annots.annotations.append(img_annot)
        img_annots.setTimestamp(frame.getTimestamp())
        self.out_annotations.send(img_annots)

    def handle_key_press(self, key: int):
        if key == ord(","):
            curr_thresh = self._current_cfg.getConfidenceThreshold()
            self._current_cfg.setConfidenceThreshold(curr_thresh - 1)
            self.out_cfg.send(self._current_cfg)
        elif key == ord("."):
            curr_thresh = self._current_cfg.getConfidenceThreshold()
            self._current_cfg.setConfidenceThreshold(curr_thresh + 1)
            self.out_cfg.send(self._current_cfg)
        elif key == ord("k"):
            all_filters = [i for i in dai.MedianFilter.__members__.values()]
            new_index = (
                all_filters.index(self._current_cfg.getMedianFilter()) + 1
            ) % len(all_filters)
            new_filter = all_filters[new_index]
            self._current_cfg.setMedianFilter(new_filter)
            self.out_cfg.send(self._current_cfg)

        elif key == ord("l"):
            curr_lr_check = self._current_cfg.getLeftRightCheck()
            self._current_cfg.setLeftRightCheck(not curr_lr_check)
            self.out_cfg.send(self._current_cfg)
