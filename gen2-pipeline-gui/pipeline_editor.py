import sys
import argparse

from Qt import QtWidgets
from NodeGraphQt import setup_context_menu
from pathlib import Path
from DAINodes import DAINodeGraph

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument( "-p", "--path", type=str, help="Path to save/load folder, relative to this script file.")
parser.add_argument( "-o", "--open", type=str, help="Path to file to open, relative to this script file")
args = vars( parser.parse_args() )

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    # create node graph controller.
    graph = DAINodeGraph()

    # set up default menu and commands.
    if args[ 'path' ]:
        filepath = str( SCRIPT_DIR / args[ 'path' ] )
    else:
        filepath = str( SCRIPT_DIR )

    if args[ 'open' ]:
        open_file = str( SCRIPT_DIR / args[ 'open' ] )
    else:
        open_file = None

    setup_context_menu( graph, set_default_file_path=filepath, open_file=open_file )

    # show the node graph widget.
    graph_widget = graph.widget
    graph_widget.show()

    app.exec_()