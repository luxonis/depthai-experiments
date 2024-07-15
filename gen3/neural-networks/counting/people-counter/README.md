# Gen2 People counting

This demo uses [person_detection_retail_0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html) neural network to detect people. The demo also displays the number of people detected on the frame.

To use a different NN model (eg. `MobileNet SSD` or `pedestrian_detection_adas_0002`), you would have to change the `size` (input size of the NN) and `nnPath` variables.

## Demo

[![image](https://user-images.githubusercontent.com/18037362/119807472-11c26580-bedb-11eb-907a-196b8bb92f28.png)]

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
