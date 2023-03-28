import record_frames_sdk
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QPushButton, QButtonGroup, QMessageBox
from PyQt5.QtGui import QIntValidator
import subprocess
import time
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

import argparse


class LiveDepthThread(QThread):
    stopped = pyqtSignal()

    def run(self):
        try:
            # Get this script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            self.live_depth_proc = subprocess.Popen(
                ["python3", f"{script_dir}/stereo_both.py"])
            self.live_depth_proc.wait()
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
                    default=['left', 'right', 'center'])
parser.add_argument('--distances', nargs='+', default=['1m', '2m', '4m'])
parser.add_argument('--resultsPath', type=str, required=False, default="./recordings",
                    help="Path to the folder containing the results to merge")
parser.add_argument('--minFileSize', type=int, required=False,
                    default=100e6, help="Minimum file size to not throw an warning")
args = parser.parse_args()


class App(QWidget):
    def __init__(self, distances, directions):
        super().__init__()
        self.distances = distances
        self.directions = directions
        self.required_vids = ["camb,c.avi", "camc,c.avi", "camd,c.avi"]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Camera Control')
        # self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        camera_id_label = QLabel('Camera ID:')
        self.camera_id_input = QLineEdit()
        self.camera_id_input.setValidator(QIntValidator())
        layout.addWidget(camera_id_label)
        layout.addWidget(self.camera_id_input)

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
            button = QRadioButton(distance)
            self.distance_buttons.append(button)
            distance_group.addWidget(button)
            self.distance_button_group.addButton(button)
        layout.addLayout(distance_group)

        self.run_button = QPushButton('Record frames')
        self.run_button.clicked.connect(self.record_frames)
        layout.addWidget(self.run_button)

        self.extra_button = QPushButton('Run live depth')
        self.extra_button.clicked.connect(self.run_live_depth)
        layout.addWidget(self.extra_button)

        self.extra_button_stop = QPushButton('Stop live depth')
        self.extra_button_stop.clicked.connect(self.stop_live_depth)
        layout.addWidget(self.extra_button_stop)
        self.extra_button_stop.setEnabled(False)

        self.transform_depth_button = QPushButton('Frames to depth')
        self.transform_depth_button.clicked.connect(
            self.transform_depth_frames)
        layout.addWidget(self.transform_depth_button)

        self.run_measurement_button = QPushButton('Run measurement')
        self.run_measurement_button.clicked.connect(
            self.run_measurement_script)
        layout.addWidget(self.run_measurement_button)

        self.setLayout(layout)

    def transform_depth_frames(self):
        self.transform_depth_button.setEnabled(False)
        QApplication.processEvents()
        try:
            camera_id, direction, distance, target_dir = self.get_test_params()
            final_dir = self.get_latest_recording_dir(target_dir)
            if not self.check_files(final_dir):
                raise RuntimeError("Not all files prenesent.")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            self.live_depth_proc = subprocess.Popen(["python3", f"{script_dir}/stereo_both.py", "-vid", "-saveFiles", "-left",
                                                    f"{final_dir}/camb,c.avi", "-right", f"{final_dir}/camc,c.avi",
                                                     "-bottom", f"{final_dir}/camd,c.avi",  "-calib", f"{final_dir}/calib.json",
                                                     "-rect", "-outVer", f"{final_dir}/outDepthVer.npy",  "-outHor",  f"{final_dir}/outDepthHor.npy",
                                                     "-outLeftRectVer", f"{final_dir}/outLeftRectVer.npy", "-outLeftRectHor",  f"{final_dir}/outLeftRectHor.npy"])
            self.live_depth_proc.wait()
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
            if not self.check_files(final_dir):
                raise RuntimeError("Not all files prenes")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            # Transform the following bash to something that looks like below.
            # Bash: $SOURCE_DIR/main.py -calib $CALIB_DIR/$cam/vermeer.json -depth $dDir/outDepthVer.npy -rectified $dDir/outLeftRectVer.npy -gt $dist -out_results_f $dDir/results_ver.txt
            # Python:
            distance = 3
            self.run_measurement_proc = subprocess.Popen(["python3", f"{script_dir}/main.py", "-calib", f"{final_dir}/calib.json", "-depth", f"{final_dir}/outDepthVer.npy",
                                                         "-rectified", f"{final_dir}/outLeftRectVer.npy", "-gt", f"{distance}", "-out_results_f", f"{final_dir}/results_ver.txt"])

            self.run_measurement_proc.wait()

            self.run_measurement_proc = subprocess.Popen(["python3", f"{script_dir}/main.py", "-calib", f"{final_dir}/calib.json", "-depth", f"{final_dir}/outDepthHor.npy",
                                                         "-rectified", f"{final_dir}/outLeftRectHor.npy", "-gt", f"{distance}", "-out_results_f", f"{final_dir}/results_hor.txt"])
            self.run_measurement_proc.wait()
        except Exception as e:
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText(f"Script failed: {str(e)}")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()

        self.transform_depth_button.setEnabled(True)

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
        if not camera_id:
            raise RuntimeError("Camera ID is not set")

        if not direction:
            raise RuntimeError("Direction is not set")

        if not distance:
            raise RuntimeError("Distance is not set")
        subdir = f"{args.resultsPath}/camera_{camera_id}/{distance}/{direction}"
        return camera_id, direction, distance, subdir

    def get_latest_recording_dir(self, target_dir):
        # Check if target dir exists
        if not os.path.isdir(target_dir):
            raise RuntimeError(f"Target dir {target_dir} does not exist")
        recordings = os.listdir(target_dir)
        latest_id = 0
        for dir in recordings:
            if "xlinkserver" not in dir:
                continue
            id = int(dir.split("-")[0])
            if id > latest_id:
                latest_id = id
        if latest_id == 0:
            raise RuntimeError("No recordings found")
        return f"{target_dir}/{latest_id}-xlinkserver"

    def record_frames(self):
        self.run_button.setEnabled(False)
        self.extra_button.setEnabled(False)
        QApplication.processEvents()
        try:
            camera_id, direction, distance, target_dir = self.get_test_params()
            # Add your custom script logic here
            # Example: custom_script(camera_id, direction, distance)
            print(
                f"Camera ID: {camera_id}, Direction: {direction}, Distance: {distance}")
            record_frames_sdk.record_frames_sdk(target_dir)

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

        self.run_button.setEnabled(True)
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
        self.run_button.setEnabled(False)
        self.extra_button.setEnabled(False)
        self.extra_button_stop.setEnabled(True)
        QApplication.processEvents()
        print("Starting live depth...")

        self.live_depth_thread = LiveDepthThread()
        self.live_depth_thread.stopped.connect(self.on_live_depth_stopped)
        self.live_depth_thread.start()

    def on_live_depth_stopped(self):
        self.run_button.setEnabled(True)
        self.extra_button.setEnabled(True)
        self.extra_button_stop.setEnabled(False)

    def stop_live_depth(self):
        print("Killing live depth...")
        self.live_depth_thread.stop_depth()


def main():
    app = QApplication(sys.argv)
    main_app = App(args.distances, args.directions)
    main_app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
