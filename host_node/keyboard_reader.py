import cv2
import depthai as dai


class KeyboardPress(dai.Buffer):
    def __init__(self, key_code: int):
        super().__init__(0)
        self._key_code = key_code

    @property
    def key(self) -> int:
        return self._key_code


class KeyboardReader(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()

    def build(self, output: dai.Node.Output) -> "KeyboardReader":
        self.link_args(output)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, _) -> None:
        key = cv2.waitKey(1)
        if key != -1:
            self.output.send(KeyboardPress(key))
