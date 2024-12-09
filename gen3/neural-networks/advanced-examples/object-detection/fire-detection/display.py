import cv2
import depthai as dai


class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._wnd_name = "video"


    def build(self, img_frames: dai.Node.Output) -> "Display":
        self.sendProcessingToPipeline(True)
        self.link_args(img_frames)
        return self
    

    def set_window_name(self, name: str) -> None:
        self._wnd_name = name


    def process(self, img_frame: dai.ImgFrame) -> None:
        frame = img_frame.getCvFrame()
        cv2.imshow(self._wnd_name, frame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
            cv2.destroyAllWindows()

    