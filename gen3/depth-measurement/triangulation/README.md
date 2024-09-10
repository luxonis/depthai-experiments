# Triangulation - Stereo neural inference demo

Because there are often application-specific host-side filtering to be done on the stereo
neural inference results, and because these calculations are lightweight
(i.e. could be done on an ESP32), we leave the triangulation itself to the host.

Demo uses [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915) neural network for detecting face and it's landmarks.

## Demo

![Stereo Inference GIF](https://user-images.githubusercontent.com/59799831/132098832-70a2d0b9-1a30-4994-8dad-dc880a803fb3.gif)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
