import depthai as dai
from depthai_nodes.utils import AnnotationHelper


class OCRAnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()

        self.input = self.createInput()
        self.passthrough = self.createInput()

        self.frame_output = self.createOutput()
        self.text_annotations_output = self.createOutput()

    def run(self):
        while self.isRunning():
            text_descriptions = self.input.get()
            passthrough_frame = self.passthrough.get()

            detections_list = text_descriptions.reference_data.detections
            recognitions_list = text_descriptions.gathered

            if len(recognitions_list) >= 1:
                annotation_helper = AnnotationHelper()

                for i, recognition in enumerate(recognitions_list):
                    detection = detections_list[i]
                    points = detection.rotated_rect.getPoints()
                    annotation_helper.draw_text(
                        "".join(recognition.classes), [points[3].x, points[3].y]
                    )

                annotations = annotation_helper.build(
                    text_descriptions.getTimestamp(), text_descriptions.getSequenceNum()
                )

            # white = np.ones_like(frame) * 255
            # # seq_num = text_descriptions.getSequenceNum()

            # # print(f"OCRAnnotation [{seq_num}] got {len(recognitions_list) } text_descriptions with ts {text_descriptions.getTimestamp()}")

            # for i, recognition in enumerate(recognitions_list):
            #     if any(recognition.scores) <= 0.75:
            #         continue
            #     detection = detections_list[i]
            #     rotated_rect = detection.rotated_rect
            #     rotated_rect = rotated_rect.denormalize(frame.shape[1], frame.shape[0])
            #     points = rotated_rect.getPoints()
            #     points = [[int(p.x), int(p.y)] for p in points]
            #     cv2.putText(
            #         white,
            #         "".join(recognition.classes),
            #         points[3],
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         1,
            #         (0, 0, 0),
            #         2,
            #     )

            # white_frame = dai.ImgFrame()
            # white_frame.setCvFrame(white, dai.ImgFrame.Type.BGR888i)
            # self.text_annotations_output.send(white_frame)

            self.frame_output.send(passthrough_frame)
            if len(recognitions_list) >= 1:
                self.text_annotations_output.send(annotations)
