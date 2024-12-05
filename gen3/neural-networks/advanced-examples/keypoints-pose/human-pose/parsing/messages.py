from typing import List

import depthai as dai


class Keypoint(dai.Buffer):
    """Keypoint class for storing a keypoint.

    Attributes
    ----------
    x: float
        X coordinate of the keypoint.
    y: float
        Y coordinate of the keypoint.
    z: Optional[float]
        Z coordinate of the keypoint.
    confidence: Optional[float]
        Confidence of the keypoint.
    """

    def __init__(self):
        """Initializes the Keypoint object."""
        super().__init__()
        self._x: float = None
        self._y: float = None
        self._z: float = 0.0
        self._confidence: float = -1.0

    @property
    def x(self) -> float:
        """Returns the X coordinate of the keypoint.

        @return: X coordinate of the keypoint.
        @rtype: float
        """
        return self._x

    @x.setter
    def x(self, value: float):
        """Sets the X coordinate of the keypoint.

        @param value: X coordinate of the keypoint.
        @type value: float
        @raise TypeError: If the X coordinate is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("x must be a float.")
        self._x = value

    @property
    def y(self) -> float:
        """Returns the Y coordinate of the keypoint.

        @return: Y coordinate of the keypoint.
        @rtype: float
        """
        return self._y

    @y.setter
    def y(self, value: float):
        """Sets the Y coordinate of the keypoint.

        @param value: Y coordinate of the keypoint.
        @type value: float
        @raise TypeError: If the Y coordinate is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("y must be a float.")
        self._y = value

    @property
    def z(self) -> float:
        """Returns the Z coordinate of the keypoint.

        @return: Z coordinate of the keypoint.
        @rtype: float
        """
        return self._z

    @z.setter
    def z(self, value: float):
        """Sets the Z coordinate of the keypoint.

        @param value: Z coordinate of the keypoint.
        @type value: float
        @raise TypeError: If the Z coordinate is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("z must be a float.")
        self._z = value

    @property
    def confidence(self) -> float:
        """Returns the confidence of the keypoint.

        @return: Confidence of the keypoint.
        @rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float):
        """Sets the confidence of the keypoint.

        @param value: Confidence of the keypoint.
        @type value: float
        @raise TypeError: If the confidence is not a float.
        @raise ValueError: If the confidence is not between 0.0 and 1.0.
        """
        if not isinstance(value, float):
            raise TypeError("confidence must be a float.")
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        self._confidence = value


class Keypoints(dai.Buffer):
    """Keypoints class for storing keypoints.

    Attributes
    ----------
    keypoints: List[dai.Keypoint]
        List of Keypoint, each representing a keypoint.
    """

    def __init__(self):
        """Initializes the Keypoints object."""
        super().__init__()
        self._keypoints: List[Keypoint] = []

    @property
    def keypoints(self) -> List[Keypoint]:
        """Returns the keypoints.

        @return: List of keypoints.
        @rtype: List[dai.Keypoint]
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: List[Keypoint]):
        """Sets the keypoints.

        @param value: List of keypoints.
        @type value: List[dai.Keypoint]
        @raise TypeError: If the keypoints are not a list.
        @raise TypeError: If each keypoint is not of type dai.Keypoint.
        """
        if not isinstance(value, list):
            raise TypeError("keypoints must be a list.")
        for item in value:
            if not isinstance(item, Keypoint):
                raise TypeError("All items in keypoints must be of type dai.Keypoint.")
        self._keypoints = value