# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    ltwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    class_name : Optional str
        Detector predicted class name.
    others : Optional any
        Other supplementary fields associated with detection that wants to be stored as a "memory" to be retrieve through the track downstream.

    Attributes
    ----------
    ltwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, ltwh, confidence, feature, class_name=None, others=None, id=-1):
        # def __init__(self, ltwh, feature):
        self.ltwh = np.asarray(ltwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.class_name = class_name
        self.others = others
        self.id = id

    def get_ltwh(self):
        return self.ltwh.copy()

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.ltwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.ltwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
