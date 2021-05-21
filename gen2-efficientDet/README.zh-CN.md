[英文文档](README.md)

# EfficientDet

您可以在下面阅读有关EfficientDet模型的更多信息 [automl's repo](https://github.com/google/automl/tree/master/efficientdet).

NN模型取自PINTO [model-zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/018_EfficientDet).
在本实验中，我们使用了 `EfficientDet-lite0`,它是最轻量的。

有关如何自行编译模型的说明:
- 下载 `lite0` 来自PINTO的压缩包 [model-zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/018_EfficientDet)
- 可以到 `FP16/myriad`  您将在其中找到IR格式的模型 (`.bin` 和 `.xml`)
- 将IR模型编译为Blob ([这里有说明](https://docs.luxonis.com/en/latest/pages/model_conversion/)). 我已经使用了在线转换器。 **请注意**，这里模型的输入层是FP16类型，您必须将其指定为 **-MyriadX编译参数** ：-ip FP16

## 演示

[![Watch the demo](https://user-images.githubusercontent.com/18037362/117892266-4c5bb980-b2b0-11eb-9c0c-68f5da6c2759.gif)](https://www.youtube.com/watch?v=UHXWj9TNGrM)

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

## 安装依赖

```
python3 -m pip install -r requirements.txt
```

如果您安装要求，则将使用 [NN performance improvements](https://github.com/luxonis/depthai-python/pull/209) 分支。 这有助于将NN FPS提高约10％。

## 用法

运行程序

```
python3 main.py
```