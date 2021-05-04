[英文文档](README.md)

# MJPEG 流服务器

该脚本允许您执行以下操作:
- 使用MJPEG流通过HTTP Server流帧
- 通过TCP服务器流数据

## 演示

[![MJPEG Streaming DepthAI](https://user-images.githubusercontent.com/5244214/90745571-92ab5b80-e2d0-11ea-9052-3f0c8afa9b0b.gif)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")


## 安装依赖

```
python3 -m pip install -r requirements.txt
```

## 用法

运行应用程序

```
python3 main.py
```

要查看流式传输的帧，请打开 [localhost:8090](http://localhost:8090)

要查看流数据，请使用

```
nc localhost 8070
```
