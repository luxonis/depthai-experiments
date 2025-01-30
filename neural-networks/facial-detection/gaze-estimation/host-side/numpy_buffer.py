import numpy as np
import depthai as dai

from utils import copy_timestamps


class NumpyBuffer(dai.Buffer):
    def __init__(self, np_data: np.ndarray, sync_origin: dai.Buffer) -> None:
        super().__init__(0)
        self.setData(np_data, sync_origin)

    def getData(self) -> np.ndarray:
        return self._np_data

    def setData(self, data: np.ndarray, sync_origin: dai.Buffer) -> None:
        self._np_data = data
        copy_timestamps(sync_origin, self)
