from imutils.video import FPS
import depthai as dai


class FPSCounter(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self._counter = FPS()
        self._name = None


    def build(self, data: dai.Node.Output) -> "FPSCounter":
        self._counter.start()
        self._data_q = data.createOutputQueue()
        return self
    

    def set_name(self, name: str) -> None:
        self._name = name


    def run(self) -> None:
        while self.isRunning():
            self._data_q.get()
            self._counter.update()


    def onStop(self) -> None:
        self._counter.stop()
        print(f"FPS {self._name}: {self._counter.fps():.2f}")