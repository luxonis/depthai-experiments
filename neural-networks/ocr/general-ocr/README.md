# Overview

We provide here an example for running a two stage text detection and OCR pipeline. This example uses PaddlePaddle [text detection](<>) and [text recognition (OCR)](https://hub.luxonis.com/ai/models/9ae12b58-3551-49b1-af22-721ba4bcf269?view=page) models from HubAI Model ZOO. The example visualizes the recognized text on an adjacent white image in the locations in which it was detected. This example showcases how a twostage pipeline can easily be built using depthai.

![Detection Output](media/highway-sign-ocr.gif)

# Instalation

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

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results by **DepthAI visualizer**
The latter runs in the browser at `http://localhost:8082`.
In case of a different client, replace `localhost` with the correct hostname.

## Example

To run the example you can simply run the following command:

```bash
python3 main.py \ 
        -d <<device ip / mxid>>
```

## Improving OCR results

This experiment is a showcase of a general OCR approach where little to no assumptions are made about the environment in which the experiment is deployed in. Implementing additional information about the deployed environment can greatly improve performance and accuracy. Some examples of possible implementation details:

- Filter out specific characters that are not expected (eg. ! " # $ * ...)
- Filter by length of recoginzed words, (eg. [license plate experiment](https://github.com/luxonis/depthai-experiments/tree/gen3/neural-networks/ocr/license-plate-recognition)). In this experiment a rudamentary lower limit of at least 2 characters per word was set [here](https://github.com/luxonis/depthai-experiments/blob/3e5e72cdb6336a924d29433a44837202f3022a69/neural-networks/ocr/general-ocr/utils/annotation_node.py#L34).
- Filter by text size, then inadequate sizes can be outright rejected.
- Filter by text score. In this experiment, a threshold score of 0.25 was set [here](https://github.com/luxonis/depthai-experiments/blob/3e5e72cdb6336a924d29433a44837202f3022a69/neural-networks/ocr/general-ocr/utils/annotation_node.py#L34).
- other improvements like filtering by color and by location in the frame.
