from Qt import QtCore, QtWidgets
from NodeGraphQt.widgets.stylesheet import *
from NodeGraphQt import BaseNode, NodeBaseWidget

class AddPortWidget(QtWidgets.QWidget):
    """
    Custom widget to be embedded inside a node.
    """

    def __init__(self, parent=None):
        super(AddPortWidget, self).__init__(parent)
        self.combo_1 = QtWidgets.QComboBox()
        self.btn_input = QtWidgets.QPushButton('< Add Input')
        self.btn_output = QtWidgets.QPushButton('Add Output >')

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setText('')
        self.name_edit.setStyleSheet(STYLE_QLINEEDIT)
        self.name_edit.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget( self.name_edit )
        layout.addWidget( self.btn_input )
        layout.addWidget( self.btn_output )


class AddPortWidgetWrapper(NodeBaseWidget):
    """
    Wrapper that allows the widget to be added in a node object.
    """

    def __init__(self, parent=None):
        super(AddPortWidgetWrapper, self).__init__(parent)

        # set the name for node property.
        self.set_name('my_widget')

        # set the label above the widget.
        self.set_label('Custom Ports')

        # set the custom widget.
        self.set_custom_widget(AddPortWidget())

        # connect up the signals & slots.
        self.wire_signals()

    def wire_signals(self):
        widget = self.get_custom_widget()

        # wire up the button.
        widget.btn_input.clicked.connect(self.on_btn_input_clicked)
        widget.btn_output.clicked.connect(self.on_btn_output_clicked)

    def on_btn_input_clicked(self):
        widget = self.get_custom_widget()
        port_name = widget.name_edit.text()
        if not port_name: return

        try:
            self.node.add_input( port_name )
        except:
            print( "Input with name " + port_name + " already exists!" )

    def on_btn_output_clicked(self):
        widget = self.get_custom_widget()
        port_name = widget.name_edit.text()
        if not port_name: return

        try:
            self.node.add_output( port_name )
        except:
            print( "Output with name " + port_name + " already exists!" )
        
    def get_value(self):
        pass

    def set_value(self, value):
        pass