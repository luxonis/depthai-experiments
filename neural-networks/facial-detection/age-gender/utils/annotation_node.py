from typing import List
import depthai as dai

from depthai_nodes import (
    ImgDetectionsExtended,
    Predictions,
    GatheredData,
    Classifications,
)
from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        gather_data_msg: dai.Node.Output,
    ) -> "AnnotationNode":
        self.link_args(gather_data_msg)
        return self

    def process(self, gather_data_msg: dai.Buffer) -> None:

        img_detections_extended_msg: ImgDetectionsExtended = (
            gather_data_msg.reference_data
        )
        assert isinstance(img_detections_extended_msg, ImgDetectionsExtended)

        age_gender_gather_data_msg_list: List[GatheredData] = gather_data_msg.gathered
        assert isinstance(age_gender_gather_data_msg_list, list)
        assert all(
            isinstance(msg, GatheredData) for msg in age_gender_gather_data_msg_list
        )

        assert len(img_detections_extended_msg.detections) == len(
            age_gender_gather_data_msg_list
        )

        annotations = AnnotationHelper()

        for img_detection_extended_msg, age_gender_gather_data_msg in zip(
            img_detections_extended_msg.detections, age_gender_gather_data_msg_list
        ):

            gender_msg: Classifications = age_gender_gather_data_msg.reference_data
            assert isinstance(gender_msg, Classifications)
            age_msg_list = age_gender_gather_data_msg.gathered
            assert isinstance(age_msg_list, list)
            assert len(age_msg_list) == 1
            age_msg: Predictions = age_msg_list[0]
            assert isinstance(age_msg, Predictions)

            xmin, ymin, xmax, ymax = (
                img_detection_extended_msg.rotated_rect.getOuterRect()
            )

            annotations.draw_rectangle(
                (xmin, ymin),
                (xmax, ymax),
            )

            annotations.draw_text(
                text=f"{gender_msg.classes[0][0]}; {int(age_msg.prediction * 100)}",
                position=(xmin, ymin),
            )

        annotations_msg = annotations.build(
            timestamp=img_detections_extended_msg.getTimestamp(),
            sequence_num=img_detections_extended_msg.getSequenceNum(),
        )

        self.out.send(annotations_msg)
