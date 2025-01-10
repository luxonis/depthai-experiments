# Gen3 People counting

This demo uses [person_detection_retail_0013](https://docs.openvino.ai/2024/omz_models_model_person_detection_retail_0013.html) neural network to detect people. The demo also displays the number of people detected on the frame.

To use a different NN model (eg. `MobileNet SSD` or `pedestrian_detection_adas_0002`), you would also have to change the `size` (input size of the NN) variable.

## Demo

\[![image](https://user-images.githubusercontent.com/18037362/119807472-11c26580-bedb-11eb-907a-196b8bb92f28.png)\]

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
