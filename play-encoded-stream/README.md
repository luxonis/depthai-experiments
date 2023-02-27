# Play Encoded Stream

These demos show how you can play H264/H265 OAK-encoded streams on the host computer.

## Usage

### Navigate to directory

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### 1. main.py using FFMPEG

It plays encoded stream by piping the output stream of the device to the input of the ffplay process.

![Encoding demo](https://user-images.githubusercontent.com/59799831/132475640-6e9f8b7f-52f4-4f75-af81-86c7f6e45b94.gif)

```
python3 main.py
```

### 2. pyav.py using PyAv library


This demo decodes encoded stream to OpenCV frames using the PyAv library. **Note** that this might freeze on Linux computers, which is due to PyAv library; see [workaround here](https://github.com/PyAV-Org/PyAV/issues/978#issuecomment-1121173652). For us, it worked as expected on Windows.

```
python3 pyav.py
```

### 3. mjpeg.py with OpenCV decoding


This demo decodes encoded MJPEG stream using OpenCV's [cv2.imdecode()](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e) function. Note that MJPEG compression isn't as great compared to H.264/H.265, more information [here](../record-replay/api/encoding_quality/).

```
python3 mjpeg.py
```

## Pre-requisites

Install requirements:
```
sudo apt install ffmpeg
python3 -m pip install -r requirements.txt
```
