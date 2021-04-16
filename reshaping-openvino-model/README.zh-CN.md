[英文文档](README.md)

# 重塑OpenVINO模型

该脚本允许您重塑OpenVINO模型输入层。

这样，如果您的模型通常接受300x300输入，则可以将尺寸更改为任意值，例如200x200

## 使用docker

```
docker build -t reshaper .
docker run -v $(pwd)/model:/home/openvino/model -e RESHAPE=200x200 -e MODEL_PATH=/home/openvino/model/face-detection-retail-0004.xml -e WEIGHTS_PATH=/home/openvino/model/face-detection-retail-0004.bin --rm reshaper
```

重塑的文件应该在`model`目录内可用


## 本地使用

```
source /opt/intel/openvino/bin/setupvars.sh  # activate OpenVINO
python3 reshape_openvino_model.py -m $(pwd)/model/face-detection-retail-0004.xml -w $(pwd)/model/face-detection-retail-0004.bin -r 200x200
```
