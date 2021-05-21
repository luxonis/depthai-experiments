[英文文档](README.md)

## [Gen2] 在DepthAI上的Deeplabv3-深度裁剪

此示例显示了如何在Gen2 API的DepthAI上运行Deeplabv3 +，并基于模型输出裁剪深度图像。

[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/18037362/116380966-5e4b5000-a80c-11eb-8013-74e2b2cb7515.gif)](https://www.youtube.com/watch?v=HiHrvJ8YtSM "Deeplabv3")

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

## 安装依赖

Install requirements
```
python3 -m pip install -r requirements.txt
```

## 用法

```
python3 main.py [-nn {path}]
```

您可以使用 `gen2-deeplabv3_person` 文件夹中的其他模型 (`mvn3` 或 `513x513` 输出)