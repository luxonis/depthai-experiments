[中文文档](README.zh-CN.md)

# Reshaping OpenVINO model

This script allows you to reshape the OpenVINO model input layer.

This way, if your model normally accepts 300x300 input, you can change the dimensions to any value, e.x. 200x200

## Using docker

```
docker build -t reshaper .
docker run -v $(pwd)/model:/home/openvino/model -e RESHAPE=200x200 -e MODEL_PATH=/home/openvino/model/face-detection-retail-0004.xml -e WEIGHTS_PATH=/home/openvino/model/face-detection-retail-0004.bin --rm reshaper
```

reshaped files should be available inside `model` directory


## Local usage

```
source /opt/intel/openvino/bin/setupvars.sh  # activate OpenVINO
python3 reshape_openvino_model.py -m $(pwd)/model/face-detection-retail-0004.xml -w $(pwd)/model/face-detection-retail-0004.bin -r 200x200
```
