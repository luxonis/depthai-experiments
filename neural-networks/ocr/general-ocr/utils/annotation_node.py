import depthai as dai
import numpy as np
import cv2


class OCRAnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()

        self.white_frame_output = self.createOutput()
        # self.text_annotations_output = self.createOutput()

    def run(self):
        while self.isRunning():
            # print("[Ann] get annotation_node/text_descriptions")
            text_descriptions = self.input.get()
            # print(f"[Ann {text_descriptions.getSequenceNum()}] got annotation_node/text_descriptions with ts {text_descriptions.getTimestamp()}")
            frame = text_descriptions["passthrough"].getCvFrame()
            detections_list = text_descriptions["detections"].detections
            recognitions_message = text_descriptions["recognitions"].recognitions

            white_frame = np.ones(frame.shape, dtype=np.uint8) * 255

            for i, recognition in enumerate(recognitions_message):
                detection = detections_list[i]
                rotated_rect = detection.rotated_rect
                rotated_rect = rotated_rect.denormalize(frame.shape[1], frame.shape[0])
                points = rotated_rect.getPoints()
                points = [[int(p.x), int(p.y)] for p in points]
                cv2.putText(
                    white_frame,
                    "".join(recognition.classes),
                    points[3],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

            # annotation = dai.ImgAnnotation()
            # img_annotations = dai.ImgAnnotations()
            # for i, recognition in enumerate(recognitions_message.recognitions):
            #     detection = detections_list[i]
            #     points = detection.rotated_rect.getPoints()

            #     text_annotation = dai.TextAnnotation()
            #     text_annotation.position = points[3]
            #     text_annotation.text = "".join(recognition.classes)
            #     text_annotation.fontSize = (points[3].y - points[0].y) * white_frame.shape[0]
            #     text_annotation.textColor = TEXT_COLOR
            #     annotation.texts.append(text_annotation)

            # img_annotations.annotations.append(annotation)
            # img_annotations.setTimestamp(recognitions_message.getTimestamp())
            output_frame = dai.ImgFrame()
            # img_annotations.setSequenceNum(text_descriptions.getSequenceNum())

            self.white_frame_output.send(
                output_frame.setCvFrame(white_frame, dai.ImgFrame.Type.NV12)
            )
            # self.text_annotations_output.send(img_annotations)
