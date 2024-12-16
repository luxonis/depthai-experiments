# Overview
We provide here an example for running a two stage text detection and OCR pipeline. This example uses PaddlePaddle text detection and text recognition (OCR) models from [HubAI Model ZOO](https://hub.luxonis.com/ai/models). The example visualizes the recognized text on an adjacent white image in the locations in which it was detected. This example showcases how a twostage pipeline can easily be built using depthai.


**WARNING:** As of depthai alpha10 the example only works on OAK4 devices. 


# Instalation
Running this example requires a **Luxonis OAK4 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:
```bash
pip install -r requirements.txt
```

# Usage
The inference is ran using a simple CLI call:
```bash
python3 main.py \
    --device ... \
    --media ...
```

The relevant arguments:
- **--device** [OPTIONAL]: DeviceID or IP of the camera to connect to.
By default, the first locally available device is used;
- **--media** [OPTIONAL]: Path to the media file to be used as input. 
Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results by **DepthAI visualizer**
The latter runs in the browser at `http://localhost:8082`.
In case of a different client, replace `localhost` with the correct hostname.

## Example
To try it out, let's run a simple YOLOv6 object detection model on your camera input.
```bash
python3 main.py \ 
        -d <<device ip / mxid>>
```