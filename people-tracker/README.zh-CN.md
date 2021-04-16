[英文文档](README.md)

# 人物追踪器

此应用程序统计视频流中有多少人向上/向下/向左/向右移动，从而使您可以接收有关多少人进入房间或​​经过走廊的信息。

在这个例子中使用的模型是[person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) 从OpenVINO模型动物园。

## 演示

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90752628-ee2d1780-e2d7-11ea-8e48-ca94b02a7674.gif)](https://youtu.be/8RiHkkGKdj0)

## 先决条件

购买DepthAI模型(请参阅 https://shop.luxonis.com/)

## 安装项目依赖

```
python3 -m pip install -r requirements.txt
```

## 运行此示例

```
python3 main.py
```

默认情况下，您应该看到一个调试窗口和控制台输出，该窗口显示了被跟踪的人数。另外，您应该看到 `results.json` 带有时间戳记的文件。

如果要在不预览的情况下运行它，只需收集数据，就可以修改 `main.py` a和设置。

```diff
- debug = True
+ debug = False
```

然后，该应用将在没有预览窗口或调试消息的情况下运行，仅将结果保存到 `results.json`

## Credits

Adrian Rosebrock, OpenCV People Counter, PyImageSearch, https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/, accessed on 6 August 2020
