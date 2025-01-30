# Speech Recognition

This example demonstrates the Gen3 Pipeline Builder running
[Whisper Tiny EN Network](https://hub.luxonis.com/ai/models/0aaf1b77-761b-44d6-893c-c473ca463186?view=page). In this demo DepthAI visualizer is integrated.

## Setup

The experiment can be run using peripheral or standalone mode.

### Peripheral mode

For peripheral mode you first need to install the required packages manually.

#### Installation

```bash
python3 -m pip install -r requirements.txt
```

For this mode only you will need to install the `depthai` and `openai-whisper` packages manually.

```bash
python3 -m pip install -U --prefer-binary --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local "depthai==3.0.0-alpha.6.dev0+4b380c003bbfe52348befdb82cf32013a7db2793"
python3 -m pip install -U openai-whisper==20231117 --no-deps
```

Furthermore, you will need to install the `ffmpeg` tool that is needed for audio pre-processing.

```bash
sudo apt-get install ffmpeg
```

#### Running

Run the application using python

```bash
python3 main.py --device_ip <device_ip> --audio_file <audio_file>
```

or using oakctl tool

```bash
oakctl run-script python3 main.py --device_ip <device_ip> --audio_file <audio_file>
```

> \[!WARNING\]
> The `--device_ip` and `--audio_file` arguments are mandatory. The `device_ip` is the IP address of the device you are connecting to. The `audio_file` is the path to the audio file you want to use for the inference.

To see the output in visualizer open browser at http://localhost:8000.

### Standalone mode

All the requirements are installed in virtual environment automatically using `oakctl` tool. Environment is setup according to the `oakapp.toml` file.

#### Connecting to the camera

Connect to the device with command

```
oakctl connect
```

If you have more cameras on your network you can list the available devices using `oakctl list` to obtain the IP adress of the desired camera.

With the obtained IP you can connect to the camera. For example desired camera has IP `192.168.0.10`, connect as

```
oakctl connect 192.168.0.10
```

OR

You can connect to the camera with it's index in the `oakctl list` table. For example if the camera is 3rd in the table, connect as

```
oakctl connect 3
```

#### Running

Run the `oakctl` app from the `whisper-tiny-en` directory as

```
oakctl app run .
```

> \[!NOTE\]
> To test the application with a different audio file, place the new file in the `whisper-tiny-en/assets/audio_files` directory. Then, update the `--audio_file` argument in the `entrypoint` field of the `oakapp.toml` file to specify the new file.

To see the output in visualizer open browser at http://192.168.0.10:8000, if `192.168.0.10` is IP of your camera.

## Using the Visualizer

Once the Visualizer is opened, click on Decoded Audio Message tab and you will see the transcribed text from the audio file.
