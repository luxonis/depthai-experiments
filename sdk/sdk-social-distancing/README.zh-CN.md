[英文文档](README.md)

# 社交隔离

在下面您可以看到一个场景中的3个人。DepthAI监视他们的3D位置并在他们小于2米`Too Close`时发出警报，在低于此阈值时显示，并始终覆盖人员之间的距离以及每个人的3D位置（以x，y，z为单位）米）。

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## 先决条件

购买DepthAI模型(请参阅 https://shop.luxonis.com/)

## 安装依赖

```
python3 -m pip install -r requirements.txt
```

## 运行示例

```
python3 main.py
```

## 这个怎么运作

![Social Distancing explanation](https://user-images.githubusercontent.com/32992551/101372410-19c51500-3869-11eb-8af4-f9b4e81a6f78.png)
