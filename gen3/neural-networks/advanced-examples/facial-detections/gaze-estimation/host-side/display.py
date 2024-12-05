import cv2
import depthai as dai


class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, img_frames: dai.Node.Output, camera: bool) -> "Display":
        self._camera = camera
        self._wnd_text = "Camera view" if camera else "Video view"
        self.link_args(img_frames)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, img_frame: dai.ImgFrame) -> None:
        frame = img_frame.getCvFrame()

        cv2.imshow(self._wnd_text, frame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
            cv2.destroyAllWindows()