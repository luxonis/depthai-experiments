import depthai as dai
from depthai_nodes.ml.messages import Keypoint

# these would be global values
OUTLINE_COLOR = dai.Color(1.0, 0.5, 0.5, 1.0)
TEXT_COLOR = dai.Color(0.5, 0.5, 1.0, 1.0)
BACKGROUND_COLOR = dai.Color(1.0, 1.0, 0.5, 1.0)

class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.inputDet = self.createInput()
        self.output = self.createOutput()
        
    def setLabelMap(self, labelMap):
        self.labelMap = labelMap
    def run(self):
        while self.isRunning():
            imgAnnt = dai.ImgAnnotations()
            annotation = dai.ImgAnnotation()
            
            nnData= self.inputDet.get()
            
            img_annotations = dai.ImgAnnotations()
            annotation = dai.ImgAnnotation()
            keypoints = []
            for keypoint in nnData.keypoints:
                keypoint: Keypoint = keypoint
                keypoints.append(dai.Point2f(keypoint.x, keypoint.y))

            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.LINE_LOOP
            pointsAnnotation.points = dai.VectorPoint2f(keypoints)
            pointsAnnotation.outlineColor = dai.Color(1.0, 0.35, 0.367, 1.0)
            pointsAnnotation.fillColor = dai.Color(1.0, 0.35, 0.367, 1.0)
            pointsAnnotation.thickness = 10.0
            annotation.points.append(pointsAnnotation)

            img_annotations.annotations.append(annotation)
            img_annotations.setTimestamp(nnData.getTimestamp())
            self.output.send(imgAnnt)