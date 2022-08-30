# Play Encoded Stream

These demos show how you can play H264/H265 OAK-encoded streams on the host computer.


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

## Pre-requisites

Install requirements:
```
sudo apt install ffmpeg
python3 -m pip install -r requirements.txt
```
