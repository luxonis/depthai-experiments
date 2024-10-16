from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import depthai as dai


class BaseMeta(ABCMeta, type(dai.node.ThreadedHostNode)):
    pass


class BaseParser(dai.node.ThreadedHostNode, metaclass=BaseMeta):
    """Base class for neural network output parsers. This class serves as a foundation
    for specific parser implementations used to postprocess the outputs of neural
    network models. Each parser is attached to a model "head" that governs the parsing
    process as it contains all the necessary information for the parser to function
    correctly. Subclasses should implement `build` method to correctly set all
    parameters of the parser and the `run` method to define the parsing logic.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    """

    def __init__(self) -> None:
        super().__init__()
        self._input = self.createInput()
        self._out = self.createOutput()

    @property
    @abstractmethod
    def input(self) -> dai.Node.Input:
        return self._input

    @property
    @abstractmethod
    def out(self) -> dai.Node.Output:
        return self._out

    @input.setter
    def input(self, node: dai.Node.Input) -> None:
        """Linking point to which the Neural Network's output is linked."""
        self._input = node

    @out.setter
    def out(self, node: dai.Node.Output) -> None:
        """Output node to which the processed network results are sent in the form of a
        DepthAI message."""
        self._out = node

    @abstractmethod
    def build(self, head_config: Dict[str, Any]) -> "BaseParser":
        """Sets the head configuration for the specified head.

        Attributes
        ----------
        head_config : Dict
            A dictionary containing configuration details relevant to the parser, including parameters and settings required for output parsing.
        """
        pass

    @abstractmethod
    def run(self):
        """Parses the output from the neural network head.

        This method should be overridden by subclasses to implement the specific parsing logic.
        It accepts arbitrary keyword arguments for flexibility.
        Args:
            **kwargs: Arbitrary keyword arguments for the parsing process.
        Returns:
            The parsed output message, as defined by the logic in the subclass.
        """
        pass