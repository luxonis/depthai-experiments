# This Python file uses the following encoding: utf-8
import argparse
import json
import sys
import time
import traceback
from functools import cmp_to_key
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtQml import QQmlApplicationEngine, qmlRegisterType
from PyQt5.QtQuick import QQuickPaintedItem
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
import depthai as dai

from PyQt5.QtWidgets import QApplication
from depthai_sdk import Previews, resizeLetterbox, PipelineManager, PreviewManager


instance = None

class ImageWriter(QQuickPaintedItem):
    frame = QImage()

    def __init__(self, parent):
        super().__init__(parent)
        self.setRenderTarget(QQuickPaintedItem.FramebufferObject)
        self.setProperty("parent", parent)

    def paint(self, painter):
        painter.drawImage(0, 0, self.frame)

    def update_frame(self, image):
        self.frame = image
        self.update()


# @QmlElement
class AppBridge(QObject):
    @pyqtSlot(str)
    def changeSelected(self, state):
        instance.guiOnPreviewChangeSelected(state)

    @pyqtSlot(bool)
    def toggleLeftRightCheck(self, state):
        instance.guiOnDepthConfigUpdate(lrc=state)

    @pyqtSlot(int)
    def setDisparityConfidenceThreshold(self, value):
        instance.guiOnDepthConfigUpdate(dct=value)

    @pyqtSlot(str)
    def setMedianFilter(self, state):
        value = getattr(dai.MedianFilter, state)
        instance.guiOnDepthConfigUpdate(median=value)


class WorkerSignals(QObject):
    updatePreviewSignal = pyqtSignal(np.ndarray)
    setDataSignal = pyqtSignal(list)
    exitSignal = pyqtSignal()


class Worker(QRunnable):
    previews = [Previews.color.name, Previews.depth.name]
    selectedPreview = Previews.color.name

    def __init__(self):
        super(Worker, self).__init__()
        self.running = False
        self.signals = WorkerSignals()
        self.signals.exitSignal.connect(self.terminate)

    def run(self):
        self.running = True

        self.pm = PipelineManager()
        self.pm.createColorCam(previewSize=(848, 480), xout=True)
        self.pm.createLeftCam(res=dai.MonoCameraProperties.SensorResolution.THE_480_P)
        self.pm.createRightCam(res=dai.MonoCameraProperties.SensorResolution.THE_480_P)
        self.pm.createDepth(useDepth=True)
        self.pm.nodes.stereo.setRuntimeModeSwitch(True)

        with dai.Device(self.pm.pipeline) as self.device:
            self.pm.createDefaultQueues(self.device)
            self.pv = PreviewManager(display=self.previews, depthConfig=dai.StereoDepthConfig(), createWindows=False)
            self.pv.createQueues(self.device)

            while self.running:
                self.pv.prepareFrames()
                self.pv.showFrames(callback=self.onShowFrame)
                time.sleep(0.01)

    def terminate(self):
        self.running = False

    def shouldRun(self):
        return self.running

    def onShowFrame(self, frame, source):
        if source == self.selectedPreview:
            self.signals.updatePreviewSignal.emit(frame)


class GuiApp:
    window = None
    threadpool = QThreadPool()

    def __init__(self):
        global instance
        self.app = QApplication([sys.argv[0]])
        self.engine = QQmlApplicationEngine()
        self.engine.quit.connect(self.app.quit)
        instance = self
        qmlRegisterType(ImageWriter, 'dai.gui', 1, 0, 'ImageWriter')
        qmlRegisterType(AppBridge, 'dai.gui', 1, 0, 'AppBridge')
        self.engine.load(str(Path(__file__).parent / "root.qml"))
        self.window = self.engine.rootObjects()[0]
        if not self.engine.rootObjects():
            raise RuntimeError("Unable to start GUI - no root objects!")

        medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))[::-1]
        self.window.setProperty("medianChoices", medianChoices)
        self.window.setProperty("previewChoices", Worker.previews)

        self.writer = self.window.findChild(QObject, "writer")
        self.previewSize = int(self.writer.width()), int(self.writer.height())

    def updatePreview(self, frame):
        scaledFrame = resizeLetterbox(frame, self.previewSize)
        if len(frame.shape) == 3:
            img = QImage(scaledFrame.data, *self.previewSize, frame.shape[2] * self.previewSize[0], 29)  # 29 - QImage.Format_BGR888
        else:
            img = QImage(scaledFrame.data, *self.previewSize, self.previewSize[0], 24)  # 24 - QImage.Format_Grayscale8
        self.writer.update_frame(img)

    def start(self):
        self.worker = Worker()
        self.worker.signals.updatePreviewSignal.connect(self.updatePreview)
        self.threadpool.start(self.worker)
        exit_code = self.app.exec()
        self.stop()
        sys.exit(exit_code)

    def stop(self):
        self.worker.signals.exitSignal.emit()
        self.worker.running = False
        self.threadpool.waitForDone(10000)

    def guiOnDepthConfigUpdate(self, median=None, dct=None, sigma=None, lrc=None, lrcThreshold=None):
        self.worker.pm.updateDepthConfig(self.worker.device, median=median, lrc=lrc)

    def guiOnPreviewChangeSelected(self, selected):
        self.worker.selectedPreview = selected
        self.selectedPreview = selected

if __name__ == "__main__":
    GuiApp().start()