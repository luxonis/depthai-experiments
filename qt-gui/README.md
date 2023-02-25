# QT GUI

This demo uses [PyQt5](https://pypi.org/project/PyQt5/) to create an interactive GUI displaying frames from DepthAI camera. 
Application allows switching between RGB and Depth previews and adjusting depth properties dynamically, like enabling Left-Right Check or changing median filtering.
Additionally, layout is built using [QML](https://doc.qt.io/qt-5/qtqml-index.html)

## Demo

![depth gif](https://user-images.githubusercontent.com/5244214/151853892-1820f30e-22cd-49a4-9a10-b20970296e4d.gif)

## Usage

### Navigate to directory

```bash
cd ./old-sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```