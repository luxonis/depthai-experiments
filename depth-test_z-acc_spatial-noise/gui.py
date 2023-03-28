import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QPushButton, QButtonGroup, QMessageBox
from PyQt5.QtGui import QIntValidator
import time

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Camera Control')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        camera_id_label = QLabel('Camera ID:')
        self.camera_id_input = QLineEdit()
        self.camera_id_input.setValidator(QIntValidator())
        layout.addWidget(camera_id_label)
        layout.addWidget(self.camera_id_input)

        self.directions = ['Left', 'Right', 'Center']
        self.direction_buttons = []
        self.direction_button_group = QButtonGroup()
        direction_group = QHBoxLayout()
        for direction in self.directions:
            button = QRadioButton(direction)
            self.direction_buttons.append(button)
            direction_group.addWidget(button)
            self.direction_button_group.addButton(button)
        layout.addLayout(direction_group)

        self.distances = ['1m', '2m', '4m']
        self.distance_buttons = []
        self.distance_button_group = QButtonGroup()
        distance_group = QHBoxLayout()
        for distance in self.distances:
            button = QRadioButton(distance)
            self.distance_buttons.append(button)
            distance_group.addWidget(button)
            self.distance_button_group.addButton(button)
        layout.addLayout(distance_group)

        self.run_button = QPushButton('Run')
        self.run_button.clicked.connect(self.run_script)
        layout.addWidget(self.run_button)

        self.setLayout(layout)


    def run_script(self):
        reply = QMessageBox.warning(self, 'Warning', "Are you sure you want to run the script?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.run_button.setEnabled(False)
            QApplication.processEvents()

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
                # Add your custom script logic here
                # Example: custom_script(camera_id, direction, distance)
                print(f"Camera ID: {camera_id}, Direction: {direction}, Distance: {distance}")
                time.sleep(2)
                raise RuntimeError("Test error")
            except Exception as e:
                error_message = QMessageBox()
                error_message.setIcon(QMessageBox.Critical)
                error_message.setWindowTitle("Error")
                error_message.setText(f"Script failed: {str(e)}")
                error_message.setStandardButtons(QMessageBox.Ok)
                error_message.exec_()
                raise e

            self.run_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    main_app = App()
    main_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()