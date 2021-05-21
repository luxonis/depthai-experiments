[英文文档](README.md)

# 人数统计

DepthAI用法的基本示例之一-在镜头前对人数进行计数并将结果保存到JSON文件（以进行进一步处理）

应用可以用作其他应用的起点或监视会议室的使用情况

本示例中使用的模型是:

- [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
- [pedestrian_detection_adas_0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html)
- [mobilenet_ssd](https://docs.openvinotoolkit.org/latest/omz_models_public_mobilenet_ssd_mobilenet_ssd.html)

## 演示

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90751105-fc7a3400-e2d5-11ea-82fe-3c7797e99e3e.gif)](https://youtu.be/M0xQI1kXju4)

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

## 安装项目依赖

```
python3 -m pip install -r requirements.txt
```

## 运行此示例

使用默认网络
```
python3 main.py
```

与特定的网络 (可以是 `person_detection_retail_0013`, `pedestrian_detection_adas_0002` 或 `mobilenet_ssd`)
```
python3 main.py -m mobilenet_ssd
```

您应该看到一个调试窗口和控制台输出，其中显示了检测到的人数。另外，您应该看到 `results.json` 带有时间戳记的文件。

如果要在不预览的情况下运行它，只需收集数据，就可以修改 `main.py` 和设置

```diff
- debug = True
+ debug = False
```

然后，该应用将在没有预览窗口或调试消息的情况下运行，仅将结果保存到 `results.json`
