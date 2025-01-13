# Age & Gender recognition

This example demonstrates how to run 2 stage inference with DepthAI library.
It recognizes age and gender of all detected faces on the frame. The demo uses [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) face detection model, crops the faces and then recognizes the age and gender of the person using [Age-Gender Model](https://hub.luxonis.com/ai/models/20cb86d9-1a4b-49e8-91ac-30f4c0a69ce1)

:exclamation **This demo currently only works on RVC4 devices**

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
To run the example you can simply run the following command:
```bash
python3 main.py \ 
        -d <<device ip / mxid>>
```