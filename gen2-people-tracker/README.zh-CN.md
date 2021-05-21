[英文文档](README.md)

# Gen2 人物追踪器

此应用程序统计视频流中向上/向下/向左/向右移动的人数，接收有关有多少人进入房间或经过走廊的信息。

该演示还可以通过SPI发送Tracklet结果。

本示例中使用的模型是 [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) 来自OpenVINO模型动物园。

## 演示

[![Watch the demo](https://user-images.githubusercontent.com/18037362/116413235-56e96e00-a82f-11eb-8007-bfcdb27d015c.gif)](https://www.youtube.com/watch?v=MHmzp--pqUA)

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

## 安装项目依赖

```
python3 -m pip install -r requirements.txt
```

## 运行示例

```
python3 main.py
```
