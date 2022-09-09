import os.path

import depthai as dai
import cv2
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QAbstractSpinBox
import os
import time

colorMode = QtGui.QImage.Format_RGB888
try:
    colorMode = QtGui.QImage.Format_BGR888
except:
    colorMode = QtGui.QImage.Format_RGB888

x_dist = 0
y_dist = 0
z_dist = 0

CSV_HEADER = "TimeStamp,MxID,Error,Result"
mx_id = None
product = None


class Ui_DepthTest(object):
    def setupUi(self, DepthTest):
        DepthTest.setObjectName("DepthTest")
        DepthTest.resize(983, 643)
        self.l_info = QtWidgets.QLabel(DepthTest)
        self.l_info.setGeometry(QtCore.QRect(20, 350, 221, 191))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.l_info.setFont(font)
        self.l_info.setStyleSheet("QLabel {\n"
                                  "    .adjust-line-height {\n"
                                  "        line-height: 1em;\n"
                                  "    }\n"
                                  "}")
        self.l_info.setObjectName("l_info")
        self.l_result = QtWidgets.QLabel(DepthTest)
        self.l_result.setGeometry(QtCore.QRect(100, 240, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.l_result.setFont(font)
        self.l_result.setObjectName("l_result")
        self.b_start = QtWidgets.QPushButton(DepthTest)
        self.b_start.setGeometry(QtCore.QRect(550, 530, 151, 71))
        self.b_start.setObjectName("b_start")
        self.l_test = QtWidgets.QLabel(DepthTest)
        self.l_test.setGeometry(QtCore.QRect(20, 40, 231, 81))
        font = QtGui.QFont()
        font.setPointSize(21)
        self.l_test.setFont(font)
        self.l_test.setObjectName("l_test")
        self.l_lidar = QtWidgets.QLabel(DepthTest)
        self.l_lidar.setGeometry(QtCore.QRect(250, 360, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.l_lidar.setFont(font)
        self.l_lidar.setObjectName("l_lidar")
        self.l_distance = QtWidgets.QLabel(DepthTest)
        self.l_distance.setGeometry(QtCore.QRect(250, 410, 71, 31))
        self.l_distance.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(18)
        font.setUnderline(False)
        self.l_distance.setFont(font)
        self.l_distance.setObjectName("l_distance")
        self.l_error = QtWidgets.QLabel(DepthTest)
        self.l_error.setGeometry(QtCore.QRect(250, 460, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.l_error.setFont(font)
        self.l_error.setObjectName("l_error")
        self.spin_manual = QtWidgets.QDoubleSpinBox(DepthTest)
        self.spin_manual.setDecimals(3)
        self.spin_manual.setRange(0.0, 9.99)
        self.spin_manual.setGeometry(QtCore.QRect(20, 140, 76, 36))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.spin_manual.setFont(font)
        self.spin_manual.setObjectName("spinManualBox")
        self.spin_manual.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.r_manual = QtWidgets.QRadioButton(DepthTest)
        self.r_manual.setGeometry(QtCore.QRect(110, 140, 201, 26))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.r_manual.setFont(font)
        self.r_manual.setObjectName("r_manual")
        self.r_sensor = QtWidgets.QRadioButton(DepthTest)
        self.r_sensor.setGeometry(QtCore.QRect(110, 190, 201, 26))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.r_sensor.setFont(font)
        self.r_sensor.setObjectName("r_sensor")
        font = QtGui.QFont()
        font.setPointSize(18)
        self.preview_video = QtWidgets.QGraphicsView(DepthTest)
        self.preview_video.setGeometry(QtCore.QRect(350, 40, 611, 471))
        self.preview_video.setObjectName("preview_video")

        self.retranslateUi(DepthTest)
        QtCore.QMetaObject.connectSlotsByName(DepthTest)
        # self.update_image()

    def retranslateUi(self, DepthTest):
        _translate = QtCore.QCoreApplication.translate
        DepthTest.setWindowTitle(_translate("DepthTest", "Depth Test"))
        self.l_info.setText(_translate("DepthTest", "<html>\n"
                                                    "<head/>\n"
                                                    "<body>\n"
                                                    "<div style=\"line-height:48px\">\n"
                                                    "<p align=\"right\">\n"
                                                    "True Distance:<br>\n"
                                                    "Detected Depth:<br>\n"
                                                    "Percentage error:</span></p>\n"
                                                    "<p align=\"center\"><span style=\" font-style:italic; "
                                                    "color:#00aa00;\">Unit: meters</span></p></div></body></html>"))
        # self.l_result.setText(_translate("DepthTest", "<html><head/><body><p><span style=\"
        # color:#00ff00;\">PASS</span></p></body></html>"))
        self.b_start.setText(_translate("DepthTest", "Start"))
        self.l_test.setText(_translate("DepthTest", "Depth Test"))
        self.l_lidar.setText(_translate("DepthTest", "0.000"))
        self.l_distance.setText(_translate("DepthTest", "0.802"))
        self.l_error.setText(_translate("DepthTest", "0.374%"))
        self.r_manual.setText(_translate("DepthTest", "Manual Distance"))
        self.r_sensor.setText(_translate("DepthTest", "Sensor Distance"))


class Camera:
    def __init__(self):
        pipeline = dai.Pipeline()
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        spatial_location_calculator = pipeline.create(dai.node.SpatialLocationCalculator)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_spatial_data = pipeline.create(dai.node.XLinkOut)
        xin_spatial_calc_config = pipeline.create(dai.node.XLinkIn)

        xout_depth.setStreamName("depth")
        xout_spatial_data.setStreamName("spatialData")
        xin_spatial_calc_config.setStreamName("spatialCalcConfig")

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        lrcheck = False
        subpixel = False

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(lrcheck)
        stereo.setSubpixel(subpixel)

        top_left = dai.Point2f(0.4, 0.4)
        bottom_right = dai.Point2f(0.6, 0.6)

        self.config = dai.SpatialLocationCalculatorConfigData()
        self.config.depthThresholds.lowerThreshold = 100
        self.config.depthThresholds.upperThreshold = 10000
        self.config.roi = dai.Rect(top_left, bottom_right)

        spatial_location_calculator.inputConfig.setWaitForMessage(False)
        spatial_location_calculator.initialConfig.addROI(self.config)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        spatial_location_calculator.passthroughDepth.link(xout_depth.input)
        stereo.depth.link(spatial_location_calculator.inputDepth)

        spatial_location_calculator.out.link(xout_spatial_data.input)
        xin_spatial_calc_config.out.link(spatial_location_calculator.inputConfig)

        self.device = dai.Device(pipeline)
        global mx_id, product
        mx_id = self.device.getDeviceInfo().getMxId()
        try:
            product = self.device.readCalibration().eepromToJson()['productMame']
        except KeyError:
            product = self.device.readCalibration().eepromToJson()['boardName']
        self.spatialCalcConfigInQueue = self.device.getInputQueue("spatialCalcConfig")
        self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.spatialCalcQueue = self.device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

        self.x_dist = 0
        self.y_dist = 0
        self.z_dist = 0

    def get_frame(self):
        in_depth = self.depthQueue.tryGet()
        if in_depth is None:
            return
        depth_frame = in_depth.getFrame()
        depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)
        depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

        spatial_data = self.spatialCalcQueue.get().getSpatialLocations()

        for depthData in spatial_data:
            if colorMode == QtGui.QImage.Format_RGB888:
                depth_frame_color = cv2.cvtColor(depth_frame_color, cv2.COLOR_BGR2RGB)

            global x_dist, y_dist, z_dist
            x_dist = depthData.spatialCoordinates.x
            y_dist = depthData.spatialCoordinates.y
            z_dist = round(depthData.spatialCoordinates.z / 1000, 2)
            return depth_frame_color

    def update_roi(self, top_left, bottom_right):
        if top_left == bottom_right:
            return
        top_left = dai.Point2f(top_left[0], top_left[1])
        bottom_right = dai.Point2f(bottom_right[0], bottom_right[1])
        self.config.roi = dai.Rect(top_left, bottom_right)
        self.config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
        config = dai.SpatialLocationCalculatorConfig()
        config.addROI(self.config)
        self.spatialCalcConfigInQueue.send(config)


class ROI:
    def __init__(self, white_rect, black_rect):
        self.whiteRect = white_rect
        self.whiteRect.setRect(0, 0, 100, 100)
        self.whiteRect.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255), 5))
        self.whiteRect.setBrush(QtGui.QBrush())

        self.blackRect = black_rect
        self.blackRect.setRect(0, 0, 100, 100)
        self.blackRect.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 4))
        self.blackRect.setBrush(QtGui.QBrush())

    def update(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        if x1 < x2:
            x = x1
        else:
            x = x2
        if y1 < y2:
            y = y1
        else:
            y = y2
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        self.whiteRect.setRect(x, y, w, h)
        self.blackRect.setRect(x, y, w, h)


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


class Frame(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, roi):
        super().__init__()
        self.pixmap = None
        self.camera = None
        self.p1 = 0, 0
        self.p2 = 0, 0
        self.acceptHoverEvents()
        self.roi = roi
        self.cameraEnabled = False
        self.width = 0
        self.height = 0

    def update_frame(self):
        if self.cameraEnabled:
            cv_frame = self.camera.get_frame()
            if cv_frame is None:
                return

            q_image = QtGui.QImage(cv_frame.data, cv_frame.shape[1], cv_frame.shape[0], colorMode)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.pixmap = pixmap.scaled(600, 600, QtCore.Qt.KeepAspectRatio)
            if self.width == 0 or self.height == 0:
                self.width = self.pixmap.width()
                self.height = self.pixmap.height()
                self.p1 = [self.width * 0.4, self.height * 0.4]
                self.p2 = [self.width * 0.6, self.height * 0.6]
                self.roi.update(self.p1, self.p2)
            self.setPixmap(self.pixmap)

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.p1 = [event.pos().x(), event.pos().y()]
        self.p1[0] = clamp(self.p1[0], 0, self.width)
        self.p1[1] = clamp(self.p1[1], 0, self.height)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.p2 = [event.pos().x(), event.pos().y()]
        self.p2[0] = clamp(self.p2[0], 0, self.width)
        self.p2[1] = clamp(self.p2[1], 0, self.height)
        self.roi.update(self.p1, self.p2)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.p2 = [event.pos().x(), event.pos().y()]
        self.p2[0] = clamp(self.p2[0], 0, self.width)
        self.p2[1] = clamp(self.p2[1], 0, self.height)
        self.roi.update(self.p1, self.p2)
        if not self.cameraEnabled:
            return
        if self.p1[0] < self.p2[0]:
            x1 = self.p1[0]
            x2 = self.p2[0]
        else:
            x1 = self.p2[0]
            x2 = self.p1[0]

        if self.p1[1] < self.p2[1]:
            y1 = self.p1[1]
            y2 = self.p2[1]
        else:
            y1 = self.p2[1]
            y2 = self.p1[1]
        top_left = x1 / self.width, y1 / self.height
        bottom_right = x2 / self.width, y2 / self.height
        self.camera.update_roi(top_left, bottom_right)

    def enable_camera(self):
        self.camera = Camera()
        self.cameraEnabled = True

    def disable_camera(self):
        global z_dist
        z_dist = 0
        self.camera.device.close()
        self.cameraEnabled = False
        if self.pixmap is not None:
            self.pixmap.fill()
        self.setPixmap(self.pixmap)


class Scene(QtWidgets.QGraphicsScene):
    def __init__(self, dialog):
        super().__init__(dialog)
        self.whiteRect = QtWidgets.QGraphicsRectItem()
        self.blackRect = QtWidgets.QGraphicsRectItem()
        self.roi = ROI(self.whiteRect, self.blackRect)
        self.frame = Frame(self.roi)
        self.addItem(self.frame)
        self.addItem(self.whiteRect)
        self.addItem(self.blackRect)

    def get_frame(self):
        return self.frame


class Application(QDialog):
    def __init__(self):
        super().__init__()
        # DepthTest = QtWidgets.QMainWindow()
        self.error = None
        self.true_distance = 0
        self.ui = Ui_DepthTest()
        self.ui.setupUi(self)
        self.scene = Scene(self)
        self.ui.preview_video.setScene(self.scene)
        # self.ui.preview_video.onm
        self.ui.b_start.clicked.connect(self.button_event)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timer_event)
        self.timer.start(1000 // 30)
        try: #TODO Find a way to scan for USB ports
            self.serial_reader = serial.Serial("/dev/ttyUSB0", 115200)
        except:
            self.serial_reader = None
        self.ui.r_sensor.setChecked(True)
        self.count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.sum = 0
        self.average_error = None
        self.set_result('')

    def set_result(self, result):
        self.ui.l_result.setText(result)
        self.ui.l_result.adjustSize()
        if result == "PASS": self.ui.l_result.setStyleSheet("color: green")
        if result == "FAIL": self.ui.l_result.setStyleSheet("color: red")

    def button_event(self):
        if self.ui.b_start.text() == "Save":
            self.scene.get_frame().disable_camera()
            self.ui.b_start.setText("Start")
            self.ui.l_test.setText('Depth Test')
            if self.pass_count < 10 and self.fail_count < 10:
                self.average_error = self.error
            self.average_error = round(self.average_error, 2)
            self.save_csv()
            self.set_result('')
        else:
            self.scene.get_frame().enable_camera()
            self.error = 0
            self.average_error = 0
            self.count = 0
            self.pass_count = 0
            self.fail_count = 0
            self.sum = 0
            self.ui.l_error.setText('None')
            self.ui.l_distance.setText('')
            self.ui.l_lidar.setText('')
            self.ui.b_start.setText("Save")
            self.ui.l_test.setText(str(product))

    def save_csv(self):
        path = os.path.realpath(__file__).rsplit('/', 1)[0] + '/depth_result/' + str(product) + '.csv'
        print(path)
        if os.path.exists(path):
            file = open(path, 'a')
        else:
            file = open(path, 'w')
            file.write(CSV_HEADER + '\n')
        if self.ui.l_result.text() != 'PASS' and self.ui.l_result.text() != 'FAIL':
            self.ui.l_result.setText('NOT TESTED')
        file.write(f'{int(time.time())},{mx_id},{self.average_error},{self.ui.l_result.text()}\n')
        file.close()

    def timer_event(self):
        # global x_dist, y_dist, z_dist
        self.scene.get_frame().update_frame()
        self.ui.l_distance.setText(f'{z_dist}')

        if self.ui.r_manual.isChecked():
            self.true_distance = self.ui.spin_manual.value()
        else:
            if self.serial_reader is not None:
                try:
                    result = ""
                    while True:
                        c = self.serial_reader.read().decode()
                        if c == '\n':
                            break
                        result += c
                    try:
                        self.true_distance = float(result[-7:-3])
                    except ValueError:
                        pass
                except serial.serialutil.PortNotOpenError:
                    pass
            else:
                try:
                    self.serial_reader = serial.Serial("/dev/ttyUSB0", 115200)
                except:
                    self.serial_reader = None
                self.true_distance = 0
        self.ui.l_lidar.setText(f'{self.true_distance}')

        if self.true_distance > 0 and z_dist > 0:
            if self.count < 30:
                self.sum += abs(self.true_distance - z_dist) * 100 / self.true_distance
                self.count += 1
            else:
                self.error = round(self.sum / 30, 2)
                self.ui.l_error.setText(f'{self.error}')
                self.count = 0
                self.sum = 0
                if self.error < 3:
                    if self.pass_count < 10:
                        self.pass_count += 1
                        self.fail_count = 0
                        if self.pass_count == 1:
                            self.average_error = self.error
                        else:
                            self.average_error += self.error / 10
                else:
                    if self.fail_count < 10:
                        self.pass_count = 0
                        self.fail_count += 1
                        if self.fail_count == 1:
                            self.average_error = self.error
                        else:
                            self.average_error += self.error / 10
        else:
            self.error = None
            self.ui.l_error.setText(f'{self.error}')
        if self.pass_count == 10:
            self.set_result('PASS')
        if self.fail_count == 10:
            self.set_result('FAIL')


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    class_instance = Application()
    class_instance.show()
    sys.exit(app.exec_())
