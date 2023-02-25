# Anomaly Detection

The goal of anomaly detection is to recognize examples of an object that may fall out of a desired distribution of acceptable objects. For example, in manufacturing, we may want to automaticaly detect when some defect occurs.

This experiment uses a [PaDiM model](https://arxiv.org/pdf/2011.08785.pdf) trained on wood. Training was done using the [anomalib](https://github.com/openvinotoolkit/anomalib) library.

![Anomaly Detection GIF](https://user-images.githubusercontent.com/60359299/199052377-aaf26332-93c2-4710-b188-704de6afcd22.gif)

## Usage

### Navigate to directory
```
cd ./api
``` 

### Install Requirements

```
pip install -r requirements.txt
```

### Run the Demo

```
python3 main.py
```
