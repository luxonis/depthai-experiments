import record_frames_sdk
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QPushButton, QButtonGroup, QMessageBox, QGroupBox, QCheckBox
from PyQt5.QtGui import QIntValidator
import subprocess
import time
import shutil
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

import argparse


class LiveDepthThread(QThread):
    stopped = pyqtSignal()

    def run(self):
        try:
            # Get this script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            subprocess.run([sys.executable, os.path.join(script_dir, "stereo_rvc3", "stereo_live.py")], check=True)

        except Exception as e:
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText(f"Script failed: {str(e)}")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()

        self.stopped.emit()

    def stop_depth(self):
        self.live_depth_proc.kill()

class LiveDepthMeasureThread(QThread):
    stopped = pyqtSignal()
    def __init__(self, vertical) -> None:
        super().__init__()
        self.vertical = vertical
    def run(self):
        try:
            additional_args = []
            if self.vertical:
                additional_args.append("-vertical")
            # Get this script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            subprocess.run([sys.executable, os.path.join(script_dir, "main.py")] + additional_args, check=True)

        except Exception as e:
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText(f"Script failed: {str(e)}")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()

        self.stopped.emit()

    def stop_depth(self):
        self.live_depth_proc.kill()

parser = argparse.ArgumentParser()
parser.add_argument('--directions', nargs='+',
                    default=['left', 'right', 'center', 'top', 'bottom'])
parser.add_argument('--distances', nargs='+', default=['1', '2', '4', '7', '15'])
parser.add_argument('--resultsPath', type=str, required=False, default="./recordings",
                    help="Path to the folder containing the results to merge")
parser.add_argument('--minFileSize', type=int, required=False,
                    default=100e6, help="Minimum file size to not throw an warning")

parser.add_argument('--device_id', type=str, required=False, default=None, help="Device id of the camera")
args = parser.parse_args()


class App(QWidget):
    def __init__(self, distances, directions):
        super().__init__()
        self.distances = distances
        self.directions = directions
        self.required_vids = ["camb,c.avi", "camc,c.avi", "camd,c.avi"]
        self.init_ui()

    def init_ui(self):
        title = "Depth testing"
        if args.device_id:
            title += f" (Device ID: {args.device_id})"
        self.setWindowTitle(title)
        # self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        camera_id_label = QLabel('Camera ID:')
        self.camera_id_input = QLineEdit()
        self.camera_id_input.setValidator(QIntValidator())
        layout.addWidget(camera_id_label)
        layout.addWidget(self.camera_id_input)


        camera_device_id_label = QLabel('Device ID (optional):')
        self.camera_device_id_input = QLineEdit()
        layout.addWidget(camera_device_id_label)
        layout.addWidget(self.camera_device_id_input)
        if args.device_id:
            self.camera_device_id_input.setEnabled(False)

        self.direction_buttons = []
        self.direction_button_group = QButtonGroup()
        direction_group = QHBoxLayout()  # Change QVBoxLayout to QHBoxLayout
        for direction in self.directions:
            button = QRadioButton(direction)
            self.direction_buttons.append(button)
            direction_group.addWidget(button)
            self.direction_button_group.addButton(button)
        layout.addLayout(direction_group)

        self.distance_buttons = []
        self.distance_button_group = QButtonGroup()
        distance_group = QHBoxLayout()  # Change QVBoxLayout to QHBoxLayout
        for distance in self.distances:
            button = QRadioButton(distance + "m")
            self.distance_buttons.append(button)
            distance_group.addWidget(button)
            self.distance_button_group.addButton(button)
        layout.addLayout(distance_group)

        # Add a new group box to contain exposure time, ISO, and record frames button
        capture_settings_group = QGroupBox('Capture Settings - if exposure time/iso is not set, auto exposure will be used')
        capture_settings_layout = QVBoxLayout()

        # Add exposure time input
        exposure_time_label = QLabel('Exposure Time:')
        self.exposure_time_input = QLineEdit()
        self.exposure_time_input.setValidator(QIntValidator())
        capture_settings_layout.addWidget(exposure_time_label)
        capture_settings_layout.addWidget(self.exposure_time_input)

        # Add ISO input
        iso_label = QLabel('ISO:')
        self.iso_input = QLineEdit()
        self.iso_input.setValidator(QIntValidator())
        capture_settings_layout.addWidget(iso_label)
        capture_settings_layout.addWidget(self.iso_input)

        # Add FPS input
        fps_label = QLabel('FPS:')
        self.fps_input = QLineEdit()
        self.fps_input.setValidator(QIntValidator())
        capture_settings_layout.addWidget(fps_label)
        capture_settings_layout.addWidget(self.fps_input)

        self.run_button1 = QPushButton('Record frames')
        self.run_button1.clicked.connect(lambda: self.record_frames(True))
        capture_settings_layout.addWidget(self.run_button1)

        self.run_button2 = QPushButton('Preview frames')
        self.run_button2.clicked.connect(lambda: self.record_frames(False))
        capture_settings_layout.addWidget(self.run_button2)

        capture_settings_group.setLayout(capture_settings_layout)
        layout.addWidget(capture_settings_group)


        # Create a new group box for "frames to depth"
        frames_to_depth_group = QGroupBox('Frames to Depth')
        frames_to_depth_layout = QVBoxLayout()

        # Add a tick box to run at the full resolution
        self.opencv_depth = QCheckBox('Use opencv for stereo depth')
        frames_to_depth_layout.addWidget(self.opencv_depth)

        # Add the existing "frames to depth" button to the group
        self.transform_depth_button = QPushButton('Frames to depth')
        self.transform_depth_button.clicked.connect(
            self.transform_depth_frames)
        frames_to_depth_layout.addWidget(self.transform_depth_button)

        self.run_measurement_button = QPushButton('Run measurement')
        self.run_measurement_button.clicked.connect(
            self.run_measurement_script)
        frames_to_depth_layout.addWidget(self.run_measurement_button)

        # Set the layout for the group box
        frames_to_depth_group.setLayout(frames_to_depth_layout)

        # Add the group box to the main layout
        layout.addWidget(frames_to_depth_group)

        # Live measurement group
        live_measurement_group = QGroupBox("Live Measurement")
        live_measurement_layout = QVBoxLayout()

        self.vertical_checkbox = QCheckBox('Vertical')
        live_measurement_layout.addWidget(self.vertical_checkbox)

        self.run_measurement_button_live = QPushButton('Run measurement (live depth)')
        self.run_measurement_button_live.clicked.connect(self.run_live_depth_measure)
        live_measurement_layout.addWidget(self.run_measurement_button_live)

        self.stop_measurement_button_live = QPushButton('Stop measurement (live depth)')
        self.stop_measurement_button_live.clicked.connect(self.stop_live_depth_measure)
        self.stop_measurement_button_live.setEnabled(False)
        live_measurement_layout.addWidget(self.stop_measurement_button_live)


        live_measurement_group.setLayout(live_measurement_layout)
        layout.addWidget(live_measurement_group)

        # Live depth group
        live_depth_group = QGroupBox("Live Depth")
        live_depth_layout = QVBoxLayout()

        self.extra_button = QPushButton('Run live depth')
        self.extra_button.clicked.connect(self.run_live_depth)
        live_depth_layout.addWidget(self.extra_button)

        self.extra_button_stop = QPushButton('Stop live depth')
        self.extra_button_stop.clicked.connect(self.stop_live_depth)
        live_depth_layout.addWidget(self.extra_button_stop)
        self.extra_button_stop.setEnabled(False)

        live_depth_group.setLayout(live_depth_layout)
        layout.addWidget(live_depth_group)

        self.setLayout(layout)

    def transform_depth_frames(self):
        self.transform_depth_button.setEnabled(False)
        QApplication.processEvents()
        opencv_depth = self.opencv_depth.isChecked()
        try:
            camera_id, direction, distance, target_dir = self.get_test_params()
            additionalParams = []
            if opencv_depth:
                additionalParams.append("-useOpenCVDepth")
            print(additionalParams)
            final_dir = self.get_latest_recording_dir(target_dir)
            if not self.check_files(final_dir):
                raise RuntimeError("Not all files prenesent.")
            script_dir = os.path.dirname(os.path.realpath(__file__))

            subprocess.run([sys.executable, os.path.join(script_dir, "stereo_rvc3/stereo_live.py"), "-saveFiles", "-videoDir",
                            final_dir,  "-calib", os.path.join(
                                final_dir, "calib.json"),
                            "-rect", "-outDir", os.path.join(final_dir, "out")] + additionalParams, check=True)

        except Exception as e:
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText(f"Script failed: {str(e)}")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()

        self.transform_depth_button.setEnabled(True)

    def run_measurement_script(self):
        self.run_measurement_button.setEnabled(False)
        QApplication.processEvents()
        try:
            camera_id, direction, distance, target_dir = self.get_test_params()
            final_dir = self.get_latest_recording_dir(target_dir)
            out_dir = Path(final_dir) / "out"
            if not self.check_files(final_dir):
                raise RuntimeError("Not all files present")
            script_dir = os.path.dirname(os.path.realpath(__file__))

            subprocess.run([sys.executable, os.path.join(script_dir, "main.py"), "-calib", os.path.join(final_dir, "calib.json"), "-depth", os.path.join(out_dir, "verticalDepth.npy"),
                            "-rectified", os.path.join(out_dir, "leftRectifiedVertical.npy"), "-gt", str(distance), "-out_results_f", os.path.join(out_dir, "results_ver.txt")], check=True)

            subprocess.run([sys.executable, os.path.join(script_dir, "main.py"), "-calib", os.path.join(final_dir, "calib.json"), "-depth", os.path.join(out_dir, "horizontalDepth.npy"),
                            "-rectified", os.path.join(out_dir, "leftRectifiedHorizontal.npy"), "-gt", str(distance), "-out_results_f", os.path.join(out_dir, "results_hor.txt")], check=True)

        except Exception as e:
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText(f"Script failed: {str(e)}")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()

        self.run_measurement_button.setEnabled(True)
        QApplication.processEvents()

    def get_test_params(self):
        camera_id = self.camera_id_input.text()

        direction = None
        for i, button in enumerate(self.direction_buttons):
            if button.isChecked():
                direction = self.directions[i].lower()

        distance = None
        for i, button in enumerate(self.distance_buttons):
            if button.isChecked():
                distance = self.distances[i]
                try:
                    distance = float(distance)
                except ValueError:
                    raise RuntimeError(f"Distance {distance} not an integer")
        if not camera_id:
            raise RuntimeError("Camera ID is not set")

        if not direction:
            raise RuntimeError("Direction is not set")

        if not distance:
            raise RuntimeError("Distance is not set")
        subdir = f"{args.resultsPath}/camera_{camera_id}/{distance}m/{direction}"
        return camera_id, direction, distance, subdir

    def get_latest_recording_dir(self, target_dir):
        # Check if target dir exists
        if not os.path.isdir(target_dir):
            raise RuntimeError(f"Target dir {target_dir} does not exist, the frames have not been recorded/transformed to depth yet.")
        recordings = os.listdir(target_dir)
        latest_id = 0
        latest_dir = None
        for dir in recordings:
            id = int(dir.split("-")[0])
            if id > latest_id:
                latest_id = id
                latest_dir = dir
        if latest_id == 0 or latest_dir is None:
            raise RuntimeError("No recordings found")
        return f"{target_dir}/{latest_dir}"

    def record_frames(self, record = False):
        self.run_button1.setEnabled(False)
        self.run_button2.setEnabled(False)
        self.extra_button.setEnabled(False)
        QApplication.processEvents()
        try:
            camera_id, direction, distance, target_dir = self.get_test_params()
            iso_num = self.iso_input.text()
            exposure_num = self.exposure_time_input.text()
            device_id = self.camera_device_id_input.text()
            fps_num = self.fps_input.text()
            if fps_num == "":
                fps_num = 10
            else:
                fps_num = int(fps_num)
            print("Test print", iso_num, exposure_num)
            # Add your custom script logic here
            # Example: custom_script(camera_id, direction, distance)
            print(f"Camera ID: {camera_id}, Direction: {direction}, Distance: {distance}m")
            if args.device_id:
                device_id = args.device_id
            if iso_num == "" or exposure_num == "":
                record_frames_sdk.record_frames_sdk(target_dir, fps=fps_num, autoExposure=True, record=record, device_id=device_id)
            else:
                print("Using manual exposure!")
                record_frames_sdk.record_frames_sdk(target_dir, fps=fps_num, autoExposure=False, iso=int(iso_num), manualExposure=int(exposure_num), record=record, device_id=device_id)

            if record:
                target_dir_final = self.get_latest_recording_dir(target_dir)
                print(f"Checking if files are in {target_dir_final}...")
                if not self.check_files(target_dir_final):
                    shutil.rmtree(target_dir_final)
                    raise RuntimeError(
                        f"Not all files were found in {target_dir_final}, was expecting {self.required_vids}. Deleting {target_dir_final}")

                sizes = self.get_file_sizes(target_dir_final, self.required_vids)
                print(f"File sizes: {sizes}")
                if min(sizes) < args.minFileSize:
                    raise RuntimeError(
                        f"File size is too small. Minimum size is {args.minFileSize} bytes, got {min(sizes)} bytes.")
        except Exception as e:
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText(f"Script failed: {str(e)}")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            print(e)

        self.run_button1.setEnabled(True)
        self.run_button2.setEnabled(True)
        self.extra_button.setEnabled(True)

    def check_files(self, subdir):
        file_list = self.required_vids
        for file in file_list:
            file_path = os.path.join(subdir, file)
            if not os.path.exists(file_path):
                return False
        return True

    def get_file_sizes(self, subdir, file_list):
        sizes = []
        for file in file_list:
            file_path = os.path.join(subdir, file)
            sizes.append(os.path.getsize(file_path))
        return sizes

    def run_live_depth(self):
        self.run_button1.setEnabled(False)
        self.run_button2.setEnabled(False)
        self.extra_button.setEnabled(False)
        self.extra_button_stop.setEnabled(True)
        QApplication.processEvents()
        print("Starting live depth...")

        self.live_depth_thread = LiveDepthThread()
        self.live_depth_thread.stopped.connect(self.on_live_depth_stopped)
        self.live_depth_thread.start()

    def on_live_depth_stopped(self):
        self.run_button1.setEnabled(True)
        self.run_button2.setEnabled(True)
        self.extra_button.setEnabled(True)
        self.extra_button_stop.setEnabled(False)

    def stop_live_depth(self):
        print("Killing live depth...")
        self.live_depth_thread.stop_depth()


    def run_live_depth_measure(self):
        self.run_button1.setEnabled(False)
        self.run_button2.setEnabled(False)
        self.extra_button.setEnabled(False)
        self.run_measurement_button_live.setEnabled(False)
        self.extra_button_stop.setEnabled(True)
        QApplication.processEvents()
        print("Starting live depth measure...")
        vertical = self.vertical_checkbox.isChecked()
        self.live_depth_thread_measure = LiveDepthMeasureThread(vertical)
        self.live_depth_thread_measure.stopped.connect(self.on_live_depth_measure_stopped)
        self.live_depth_thread_measure.start()

    def on_live_depth_measure_stopped(self):
        self.run_button1.setEnabled(True)
        self.run_button2.setEnabled(True)
        self.extra_button.setEnabled(True)
        self.run_measurement_button_live.setEnabled(True)
        self.stop_measurement_button_live.setEnabled(False)

    def stop_live_depth_measure(self):
        print("Killing live depth...")
        self.live_depth_thread_measure.stop_depth()
def main():
    app = QApplication(sys.argv)
    main_app = App(args.distances, args.directions)
    main_app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
