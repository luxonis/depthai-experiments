import sys
import depthai as dai
from pathlib import Path
import cv2
import numpy as np
from host_qt_output import QTOutput

from PyQt5.QtQml import QQmlApplicationEngine, qmlRegisterType
from PyQt5.QtQuick import QQuickPaintedItem
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5.QtWidgets import QApplication

instance = None

def resizeLetterbox(frame: np.ndarray, size: tuple[int]):
    # Transforms the frame to meet the desired size, preserving the aspect ratio and adding black borders (letterboxing)
    border_v = 0
    border_h = 0
    if (size[1] / size[0]) >= (frame.shape[0] / frame.shape[1]):
        border_v = int((((size[1] / size[0]) * frame.shape[1]) - frame.shape[0]) / 2)
    else:
        border_h = int((((size[0] / size[1]) * frame.shape[0]) - frame.shape[1]) / 2)
    frame = cv2.copyMakeBorder(frame, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    return cv2.resize(frame, size)


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
    previews = ["color", "depth"]
    selectedPreview = "color"

    def __init__(self):
        super(Worker, self).__init__()
        self.running = False
        self.signals = WorkerSignals()

        self.config_queue = None
        self.stereo_config = dai.StereoDepthConfig()
        self.output = None

    def run(self):
        self.running = True

        with dai.Pipeline() as pipeline:

            print("Creating pipeline...")
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam.setPreviewSize(848, 480)
            cam.setInterleaved(False)
            cam.initialControl.setManualFocus(130)

            left = pipeline.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            right = pipeline.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            stereo.initialConfig.setLeftRightCheck(False)
            stereo.initialConfig.setConfidenceThreshold(255)
            stereo.initialConfig.setSubpixelFractionalBits(3)
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            stereo.setRuntimeModeSwitch(True)

            self.stereo_config.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            self.stereo_config.setLeftRightCheck(False)
            self.stereo_config.setConfidenceThreshold(255)
            self.stereo_config.setSubpixelFractionalBits(3)
            self.config_queue = stereo.inputConfig.createInputQueue()

            self.output = pipeline.create(QTOutput).build(
                preview=cam.preview,
                stereo=stereo,
                show_callback=self.onShowFrame
            )

            print("Pipeline created.")
            pipeline.run()

    def terminate(self):
        self.output.terminate()
        self.running = False

    def shouldRun(self):
        return self.running

    def onShowFrame(self, frame, source):
        if self.signals and self.running and source == self.selectedPreview:
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

        self.app.aboutToQuit.connect(self.close_application)
        sys.exit(self.app.exec())

    def close_application(self):
        self.worker.terminate()
        self.threadpool.waitForDone(1000)
        self.app.exit()

    def guiOnDepthConfigUpdate(self, median=None, dct=None, lrc=None):
        if self.worker.running:
            if median is not None:
                self.worker.stereo_config.setMedianFilter(median)
            if dct is not None:
                self.worker.stereo_config.setConfidenceThreshold(dct)
            if lrc is not None:
                self.worker.stereo_config.setLeftRightCheck(lrc)

            self.worker.config_queue.send(self.worker.stereo_config)

    def guiOnPreviewChangeSelected(self, selected):
        if self.worker.running:
            self.worker.selectedPreview = selected
            self.selectedPreview = selected

if __name__ == "__main__":
    GuiApp().start()