import depthai as dai
import numpy as np
import cv2
from depthai_nodes import SegmentationMask
from depthai_nodes import ImgDetectionsExtended


class SegAnnotationNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self, image_frame_msg: dai.Node.Output, segmentation_msg: dai.Node.Output
    ):
        self.link_args(image_frame_msg, segmentation_msg)
        return self

    def process(self, image_frame_msg: dai.Buffer, segmentation_msg: dai.Buffer):
        assert isinstance(image_frame_msg, dai.ImgFrame)
        assert isinstance(segmentation_msg, SegmentationMask)

        frame = image_frame_msg.getCvFrame()
        output_frame = dai.ImgFrame()

        mask = segmentation_msg.mask
        unique_values = np.unique(mask[mask >= 0])
        scaled_mask = np.zeros_like(mask, dtype=np.uint8)

        if unique_values.size != 0:
            min_val, max_val = unique_values.min(), unique_values.max()

            if min_val == max_val:
                scaled_mask = np.ones_like(mask, dtype=np.uint8) * 255
            else:
                scaled_mask = ((mask - min_val) / (max_val - min_val) * 255).astype(
                    np.uint8
                )
            scaled_mask[mask == -1] = 0
        colored_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_RAINBOW)
        colored_mask[mask == -1] = [0, 0, 0]

        frame_height, frame_width, _ = frame.shape
        colored_mask = cv2.resize(
            colored_mask, (frame_width, frame_height), interpolation=cv2.INTER_AREA
        )

        colored_frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

        self.output.send(
            output_frame.setCvFrame(colored_frame, dai.ImgFrame.Type.BGR888i)
        )


class DetSegAnntotationNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(self, image_frame_msg: dai.Node.Output, detections_msg: dai.Node.Output):
        self.link_args(image_frame_msg, detections_msg)
        return self

    def process(self, image_frame_msg: dai.Buffer, detections_msg: dai.Buffer):
        assert isinstance(image_frame_msg, dai.ImgFrame)
        assert isinstance(detections_msg, ImgDetectionsExtended)

        time_stamp = image_frame_msg.getTimestamp()
        frame = image_frame_msg.getCvFrame()
        output_frame = dai.ImgFrame()

        label_mask = detections_msg.masks
        detections = detections_msg.detections
        if len(label_mask.shape) < 2:
            self.out.send(output_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888i))
        else:
            detection_labels = {
                idx: detection.label for idx, detection in enumerate(detections)
            }
            detection_labels[-1] = -1

            if len(detection_labels) > 0:
                label_mask = np.vectorize(lambda x: detection_labels.get(x, -1))(
                    label_mask
                )

            color_mask = label_mask.copy()
            color_mask[label_mask == -1] = 0
            color_mask = color_mask.astype(np.uint8)
            color_mask = cv2.applyColorMap(color_mask, cv2.COLORMAP_HSV)
            color_mask[label_mask == -1] = frame[label_mask == -1]

            colored_frame = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)

            output_frame.setTimestamp(time_stamp)
            self.output.send(
                output_frame.setCvFrame(colored_frame, dai.ImgFrame.Type.BGR888i)
            )
