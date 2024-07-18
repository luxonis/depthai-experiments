## CREStereo

CREStereo: Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation

## Demo

### 160x240

>```crestereo_init_iter2_160x240_6_shaves.blob```

![demo-gif](https://i.imgur.com/S4BElZo.png)


### 120x160

>```crestereo_init_iter2_120x160_6_shaves.blob```

![demo-gif](https://i.imgur.com/4Gpt2On.png)

## Installation

```
python3 -m pip install -r requirements.txt
python3 download.py
```

## Usage

Run the application

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -nn {120x160,160x240}, --nn-choice {120x160,160x240}
                        Choose between 2 neural network models from {120x160,160x240} (the bigger one is default)
```

## References

* [ibaiGorordo/CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch)
* [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
* [Paper](https://arxiv.org/abs/2203.11483)