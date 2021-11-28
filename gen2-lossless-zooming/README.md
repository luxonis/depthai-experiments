# Lossless Zooming

This demo shows how you can achieve lossless zooming on the device. Demo will zoom into the first face it detects. It will crop 4K frames into 1080P, centered around the face.

## Demo

![Lossless Zooming]()


## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

To see the streamed frames, open [localhost:8090](http://localhost:8090).  This works in Chrome, but not Firefox or Safari.

To see the streamed data, use

```
nc localhost 8070
```
