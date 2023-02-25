## Anomalib Model Conversion

### Environment

Export requires the use of [anomalib](https://github.com/openvinotoolkit/anomalib). The conversion was only tested on commit `2f0a87cfb874d129c93a78de479e90630fbecde0`.

Install the following requirements, as well as [PyTorch](https://pytorch.org/get-started/locally/).
```
pip install anomalib
pip install openvino-dev==2021.4.2
pip install blobconverter==1.3.0
```

### Training

To train your own model or with your own data, follow the instructions [here](https://github.com/openvinotoolkit/anomalib#training).

### Export

Once ready to export, copy the script here to anomalib
```
cp ./export.py /path/to/anomalib
```

Then, you can call the script with the following options and files will be saved to the `export` directory.
```
python3 export.py --checkpoint /path/to/checkpoint.ckpt --name my_model
```
