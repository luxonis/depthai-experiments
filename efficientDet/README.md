# EfficientDet

You can read more about the EfficientDet model in [automl's repo](https://github.com/google/automl/tree/master/efficientdet).

The NN model is taken from PINTOs [model-zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/018_EfficientDet).
In this experiment we have used `EfficientDet-lite0`, which is the most lightweight one.

Instructions on how to compile the model yourself:
- Download the `lite0` zip from PINTO's [model-zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/018_EfficientDet)
- Navigate to `FP16/myriad` where you will find model in IR format (`.bin` and `.xml`)
- Compile the IR model into blob ([instructions here](https://docs.luxonis.com/en/latest/pages/model_conversion/)). I have used the online converter. **Note** here that the model's input layer is of type FP16 and you have to specify that as **MyriadX compile params**: `-ip FP16`

## Demo

[![Watch the demo](https://user-images.githubusercontent.com/18037362/117892266-4c5bb980-b2b0-11eb-9c0c-68f5da6c2759.gif)](https://www.youtube.com/watch?v=UHXWj9TNGrM)

## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```