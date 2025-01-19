## License Plate Recognition

This example demonstrates how to run 3 stage inference on DepthAI
First, a vehicle is detected on the image, the cropped image is then fed into a license plate detection model. The cropped license plate is sent to a text recognition (OCR) network,
which tries to decode the license plates texts.

**:exclamation: Due to the high comutational cost, this example only works on OAK4 devices. :exclamation:**

Take a look at [How to Train and Deploy a License Plate Detector to the Luxonis OAK](https://blog.roboflow.com/oak-deploy-license-plate/) tutorial for training a custom detector using the Roboflow platform.

## Demo
![Detection Output](visualizations/output.gif)

## Instalation

Running this example requires a **Luxonis OAK4 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

The inference is ran using a simple CLI call:

```bash
python3 main.py \
    --device ... \
    --media ...
```

The relevant arguments:

- **--device** \[OPTIONAL\]: DeviceID or IP of the camera to connect to.
  By default, the first locally available device is used;
- **--media** \[OPTIONAL\]: Path to the media file to be used as input.
  Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
  By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results by **Depthai Visualizer**

### Example

To run the example you can simply run the following command:

```bash
python3 main.py \ 
        -d <<device ip / mxid>>
```
