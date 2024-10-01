import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended, ImgDetectionsExtended


def img_detection_to_points(
    img_detection: dai.ImgDetection | ImgDetectionExtended,
) -> dict[str, dai.Point2f]:
    return {
        "top_left": dai.Point2f(img_detection.xmin, img_detection.ymin),
        "bottom_right": dai.Point2f(img_detection.xmax, img_detection.ymax),
    }


def img_detections_to_points(
    img_detections: dai.ImgDetections | ImgDetectionsExtended,
) -> list[dict[str, dai.Point2f]]:
    return [img_detection_to_points(i) for i in img_detections.detections]


def points_to_spatial_img_detection(
    points: dict[str, dai.Point3f],
    confidence: float,
    label: int,
) -> dai.SpatialImgDetection:
    spatial_img_detection = dai.SpatialImgDetection()
    spatial_img_detection.xmin = points["top_left"].x
    spatial_img_detection.ymin = points["top_left"].y
    spatial_img_detection.xmax = points["bottom_right"].x
    spatial_img_detection.ymax = points["bottom_right"].y
    spatial_img_detection.spatialCoordinates.x

    center_x = (points["top_left"].x + points["bottom_right"].x) / 2
    center_y = (points["top_left"].y + points["bottom_right"].y) / 2
    center_z = (points["top_left"].z + points["bottom_right"].z) / 2
    spatial_img_detection.spatialCoordinates = dai.Point3f(center_x, center_y, center_z)

    spatial_img_detection.confidence = confidence
    spatial_img_detection.label = label

    return spatial_img_detection


def points_to_spatial_img_detections(
    points: dict[str, dai.Point3f], confidences: list[float], labels: list[int]
) -> dai.SpatialImgDetections:
    detections = dai.SpatialImgDetections()
    detections.detections = [
        points_to_spatial_img_detection(points, conf, label)
        for points, conf, label in zip(points, confidences, labels)
    ]
    return detections
