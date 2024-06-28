import QtQuick 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1


import dai.gui 1.0

ApplicationWindow {
    width: 900
    height: 500
    Material.theme: Material.Dark
    Material.accent: Material.Red
    visible: true

    property var previewChoices
    property var medianChoices

    AppBridge {
        id: appBridge
    }

    Rectangle {
        id: root
        x: 0
        y: 0
        width: parent.width
        height: parent.height
        color: "#000000"
        enabled: true

        Rectangle {
            id: cameraPreviewRect
            color: "black"
            width: 640
            height: parent.height

            ComboBox {
                id: comboBoxImage
                x: 100
                y: 5
                width: 150
                height: 30
                model: previewChoices
                onActivated: function(index) {
                    appBridge.changeSelected(model[index])
                }
            }

            ImageWriter {
                id: imageWriter
                objectName: "writer"
                x: 10
                y: 60
                width: 600
                height: parent.height - 100
            }
        }


        Rectangle {
            id: propsRect
            color: "black"
            x: 640
            width: 360
            height: parent.height

            ComboBox {
                id: comboBox
                x: 0
                y: 102
                width: 195
                height: 33
                model: medianChoices
                onActivated: function(index) {
                    appBridge.setMedianFilter(model[index])
                }
            }

            Slider {
                id: dctSlider
                x: 360
                y: 89
                width: 200
                height: 25
                snapMode: RangeSlider.NoSnap
                stepSize: 1
                from: 0
                to: 255
                value: 240
                onValueChanged: {
                    appBridge.setDisparityConfidenceThreshold(value)
                }
            }

            Text {
                id: text2
                x: 0
                y: 71
                width: 195
                height: 25
                color: "#ffffff"
                text: qsTr("Median filtering")
                font.pixelSize: 18
                font.styleName: "Regular"
                font.weight: Font.Medium
                font.family: "Courier"
            }

            Switch {
                id: switch1
                x: 0
                y: 187
                text: qsTr("<font color=\"white\">Left Right Check</font>")
                transformOrigin: Item.Center
                font.family: "Courier"
                autoExclusive: false
                onToggled: {
                    appBridge.toggleLeftRightCheck(switch1.checked)
                }
            }
        }
    }
}
