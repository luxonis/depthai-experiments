import depthai as dai
from depthai_nodes.ml.helpers.constants import TEXT_COLOR, OUTLINE_COLOR


class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()

    def run(self):
        while self.isRunning():
            descriptions = self.input.get()

            detections_msg = descriptions["detections"]
            genders_msg = descriptions["genders"]
            ages_msg = descriptions["ages"]

            annotation = dai.ImgAnnotation()
            img_annotations = dai.ImgAnnotations()
            for detection, gender, age in zip(
                detections_msg.detections, genders_msg.genders, ages_msg.ages
            ):
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
                    f"{gender.classes[0]} {int(age.prediction * 100)}"
                )
                text_annotation.fontSize = 50
                text_annotation.textColor = TEXT_COLOR
                annotation.texts.append(text_annotation)

            img_annotations.annotations.append(annotation)
            img_annotations.setTimestamp(descriptions.getTimestamp())
            img_annotations.setSequenceNum(descriptions.getSequenceNum())

            self.output.send(img_annotations)
