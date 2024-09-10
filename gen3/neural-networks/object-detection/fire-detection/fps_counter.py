from imutils.video import FPS
import depthai as dai


class FPSCounter(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self._counter = FPS()
        self._name = None


    def build(self, data: dai.Node.Output) -> "FPSCounter":
        self._counter.start()
        self.link_args(data)
        return self
    

    def setName(self, name: str) -> None:
        self._name = name


    def process(self, _: dai.Buffer) -> None:
        self._counter.update()


    def onStop(self) -> None:
        self._counter.stop()
        print(f"FPS {self._name}: {self._counter.fps():.2f}")