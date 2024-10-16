from typing import Any, Dict

import depthai as dai
import numpy as np

from .create_keypoints import create_keypoints_message
from .base_parser import BaseParser


class KeypointParser(BaseParser):
    """Parser class for 2D or 3D keypoints models. It expects one ouput layer containing
    keypoints. The number of keypoints must be specified. Moreover, the keypoints are
    normalized by a scale factor if provided.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_name: str
        Name of the output layer from which the scores are extracted.
    scale_factor : float
        Scale factor to divide the keypoints by.
    n_keypoints : int
        Number of keypoints the model detects.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing 2D or 3D keypoints.

    Error Handling
    --------------
    **ValueError**: If the number of keypoints is not specified.

    **ValueError**: If the number of coordinates per keypoint is not 2 or 3.

    **ValueError**: If the number of output layers is not 1.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        scale_factor: float = 1.0,
        n_keypoints: int = None,
    ) -> None:
        """Initializes KeypointParser node.

        @param output_layer_name: Name of the output layer from which the keypoints are
            extracted.
        @type output_layer_name: str
        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.scale_factor = scale_factor
        self.n_keypoints = n_keypoints

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "KeypointParser":
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        KeypointParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 1:
            raise ValueError(
                f"Only one output layer supported for Keypoint, got {output_layers} layers."
            )
        self.output_layer_name = output_layers[0]
        self.scale_factor = head_config.get("scale_factor", self.scale_factor)
        self.n_keypoints = head_config.get("n_keypoints", self.n_keypoints)

        return self

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the name of the output layer.

        @param output_layer_name: The name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name

    def setScaleFactor(self, scale_factor: float) -> None:
        """Sets the scale factor to divide the keypoints by.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        if not isinstance(scale_factor, float):
            raise ValueError("Scale factor must be a float.")
        self.scale_factor = scale_factor

    def setNumKeypoints(self, n_keypoints: int) -> None:
        """Sets the number of keypoints.

        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        """
        if not isinstance(n_keypoints, int):
            raise ValueError("Number of keypoints must be an integer.")
        self.n_keypoints = n_keypoints

    def run(self):
        if self.n_keypoints is None:
            raise ValueError("Number of keypoints must be specified!")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            layers = output.getAllLayerNames()

            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            keypoints = output.getTensor(
                self.output_layer_name, dequantize=True
            ).astype(np.float32)
            num_coords = int(np.prod(keypoints.shape) / self.n_keypoints)

            if num_coords not in [2, 3]:
                raise ValueError(
                    f"Expected 2 or 3 coordinates per keypoint, got {num_coords}."
                )

            keypoints = keypoints.reshape(self.n_keypoints, num_coords)

            keypoints /= self.scale_factor

            msg = create_keypoints_message(keypoints)
            msg.setTimestamp(output.getTimestamp())

            self.out.send(msg)