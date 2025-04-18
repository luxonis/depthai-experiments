import depthai as dai
from depthai_nodes import PRIMARY_COLOR


class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()

    def run(self):
        while self.isRunning():
            landmarks = self.input.get()
            detections_msg = landmarks["detections"]
            gazes_msg = landmarks["gazes"]

            annotation = dai.ImgAnnotation()
            img_annotations = dai.ImgAnnotations()
            src_w, src_h = detections_msg.transformation.getSourceSize()

            for detection, gaze in zip(detections_msg.detections, gazes_msg.landmarks):
                face_bbox = detection.rotated_rect.getPoints()
                keypoints = detection.keypoints

                points_annotation = dai.PointsAnnotation()
                points_annotation.type = dai.PointsAnnotationType.LINE_LOOP
                points_annotation.points = dai.VectorPoint2f(face_bbox)
                points_annotation.outlineColor = PRIMARY_COLOR
                points_annotation.thickness = 2
                annotation.points.append(points_annotation)

                # Draw gaze
                gaze_tensor = gaze.getTensor("Identity", dequantize=True)
                gaze_tensor = gaze_tensor.flatten()

                left_eye = keypoints[0]
                le_line_ann = self._draw_line(
                    dai.Point2f(left_eye.x, left_eye.y), gaze_tensor, src_w, src_h
                )
                annotation.points.append(le_line_ann)

                right_eye = keypoints[1]
                re_line_ann = self._draw_line(
                    dai.Point2f(right_eye.x, right_eye.y), gaze_tensor, src_w, src_h
                )
                annotation.points.append(re_line_ann)

            img_annotations.annotations.append(annotation)
            img_annotations.setTimestamp(landmarks.getTimestamp())
            img_annotations.setSequenceNum(landmarks.getSequenceNum())

            self.output.send(img_annotations)

    def _draw_line(
        self, start_point: dai.Point2f, vector: list, src_w: int, src_h: int
    ) -> dai.PointsAnnotation:
        gaze_vector = (vector * 640)[:2]
        gaze_vector_x = gaze_vector[0] / src_w
        gaze_vector_y = gaze_vector[1] / src_h
        end_point = dai.Point2f(
            start_point.x + gaze_vector_x, start_point.y - gaze_vector_y
        )

        line_ann = dai.PointsAnnotation()
        line_ann.type = dai.PointsAnnotationType.LINE_STRIP
        line_ann.points = dai.VectorPoint2f([start_point, end_point])
        line_ann.outlineColor = dai.Color(0.0, 0.0, 1.0, 1.0)
        line_ann.thickness = 2
        return line_ann
