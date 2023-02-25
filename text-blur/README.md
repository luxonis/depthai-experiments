##  Text blurring on DepthAI

This example shows an implementation of [Text Detection](https://github.com/MhLiao/DB) on DepthAI in the Gen2 API system with additional text blurring. ONNX is taken from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/145_text_detection_db), and reexported with preprocessing flags.

Supported input shape is 480 x 640 (H x W), with expected ~3 FPS.

![Image example](assets/example.gif)

Cat image is taken from [here](https://www.pexels.com/photo/grey-kitten-on-floor-774731/), dog image is taken from [here](https://www.pexels.com/photo/brown-and-white-american-pit-bull-terrier-with-brown-costume-825949/).

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py [options]
```

Options:

* `-nn, --nn_model`: Select model path for inference. Default: *models/text_detection_db_256x320_openvino_2021.4_6shave.blob*.
* `-bt, --box_thresh`: Set the box confidence threshold. Default: *0.2*.
* `-t, --thresh`: Set the bitmap threshold. Default: *0.1*.
* `-ms, --min_size`: Set the minimum size of box (area). Default: *1*.
* `-mc, --max_candidates`: Maximum number of returned box candidates. Default: *75*.
