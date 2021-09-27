[英文文档](README.md)

# Two Stage Inference

该示例展示了如何使用DepthAI执行两阶段推理。

我们将使用`face-detection-retail-0004`检测脸部并将其`landmarks-regression-retail-0009`作为第二阶段推断，该步骤将检测检测到的脸部上的脸部界标

## 安装依赖

```
python3 -m pip install -r requirements.txt
```

## 用法

运行应用程序

```
python3 main.py
```