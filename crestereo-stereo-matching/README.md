# CREStereo

CREStereo: Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation

## Demo

### 160x240

>```python3 main.py -nn models/crestereo_init_iter2_160x240_6_shaves.blob -shape 160x240```

![demo-gif](https://i.imgur.com/S4BElZo.png)


### 120x160

>```python3 main.py -nn models/crestereo_init_iter2_120x160_6_shaves.blob -shape 120x160```

![demo-gif](https://i.imgur.com/4Gpt2On.png)

## Usage

### Navigate to directory
```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

Download blobs:
```
python3 download.py
```

### Launch the script

```
python3 main.py -nn NN_PATH -shape 120x160
``` 
> Note: Make sure `-shape` matches the blob's model input shape

Arguments:
```
usage: main.py [-h] -nn NN_PATH -shape {120x160,160x240}

optional arguments:
  -h, --help            show this help message and exit
  -nn NN_PATH, --nn_path NN_PATH
                        select model blob path for inference
  -shape {120x160,160x240}, --shape {120x160,160x240}
                        model input shape, same as used blob
```

# References
* [ibaiGorordo/CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch)
* [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
* [Paper](https://arxiv.org/abs/2203.11483)