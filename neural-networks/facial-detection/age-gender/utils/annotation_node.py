import depthai as dai
from depthai_nodes import TEXT_COLOR, OUTLINE_COLOR, DetectedRecognitions
from typing import List

class AnnotationNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput()

    def build(
        self,
        det_age_recognitions: dai.Node.Output,
        det_gender_recognitions: dai.Node.Output,
    ) -> "AnnotationNode":
        self.link_args(det_age_recognitions, det_gender_recognitions)
        return self

    def process(self, det_age_recognitions: dai.Buffer, det_gender_recognitions: dai.Buffer) -> None:
        assert isinstance(det_age_recognitions, DetectedRecognitions)
        assert isinstance(det_gender_recognitions, DetectedRecognitions)

        detections_list: List[dai.ImgDetection] = det_age_recognitions.img_detections.detections

        age_msg = det_age_recognitions.nn_data
        genders_msg = det_gender_recognitions.nn_data

        annotation = dai.ImgAnnotation()
        img_annotations = dai.ImgAnnotations()
        for detection, gender, age in zip(
            detections_list, genders_msg, age_msg
        ):
            print("gender", type(gender), "age", type(age))
            points = detection.rotated_rect.getPoints()

            points_annotation = dai.PointsAnnotation()
            points_annotation.type = dai.PointsAnnotationType.LINE_LOOP
            points_annotation.points = dai.VectorPoint2f(points)
            points_annotation.outlineColor = OUTLINE_COLOR
            points_annotation.thickness = 2
            annotation.points.append(points_annotation)

            text_annotation = dai.TextAnnotation()
            text_annotation.position = points[0]
            text_annotation.text = (
                f"{gender.classes[0]} {int(age.predictions[0].prediction * 100)}"
            )
            print(f"{gender.classes[0]} {int(age.predictions[0].prediction * 100)}")
            text_annotation.fontSize = 50
            text_annotation.textColor = TEXT_COLOR
            annotation.texts.append(text_annotation)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(det_age_recognitions.getTimestamp())
        img_annotations.setSequenceNum(det_age_recognitions.getSequenceNum())

        self.output.send(img_annotations)
