import cv2
import depthai as dai
import numpy as np


class DensityMapToFrame(dai.node.HostNode):
    """A host node that receives density map and transforms it into an image frame.

    Attributes
    ----------
    output : dai.ImgFrame
        The output message for the density map overlayed frame.
    """

    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.setColormap(cv2.COLORMAP_HOT)

    def setColormap(self, colormap_value: int) -> None:
        color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
        color_map[0] = [0, 0, 0]
        self._colormap = color_map

    def build(self, map_msg: dai.Node.Output) -> "DensityMapToFrame":
        self.link_args(map_msg)
        return self

    def process(self, map_msg: dai.Buffer) -> None:
        density_map = map_msg.map
        density_map_normalized = (
            density_map / density_map.max() if density_map.max() > 0 else density_map
        )
        density_map_image = cv2.applyColorMap(
            (density_map_normalized * 255).astype(np.uint8), self._colormap
        )

        density_map_frame_msg = dai.ImgFrame()
        density_map_frame_msg.setCvFrame(density_map_image, dai.ImgFrame.Type.BGR888i)
        density_map_frame_msg.setTimestamp(map_msg.getTimestamp())
        density_map_frame_msg.setSequenceNum(map_msg.getSequenceNum())
        self.output.send(density_map_frame_msg)
