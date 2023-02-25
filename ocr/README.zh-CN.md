[英文文档](README.md)

##  文本检测+光学字符识别（OCR）管道

该管道执行文本检测（EAST），然后对检测到的文本进行光学字符识别。它实现了问题 [#124](https://github.com/luxonis/depthai/issues/124).

## 怎么跑

本示例使用 [Gen2 Pipeline Builder](https://github.com/luxonis/depthai/issues/136), 因此请确保安装以下要求以确保适当的Gen2 API版本。

`python3 -m pip install -r requirements.txt`

安装需求后，运行此管道非常简单:

`python3 main.py`

## 示例结果

运行后，您可以将其指向感兴趣的文本，以获取检测结果以及检测到的区域（以及文本在像素空间中的位置）中的结果文本。

请注意，框架中的文本越多，网络运行的速度就越慢-因为它在每个检测到的文本区域上都运行OCR。 
您将在以下示例中看到这种差异:


[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749743-13febe00-5f01-11eb-8b5f-dca801f5d125.png)](https://www.youtube.com/watch?v=Bv-p76A3YMk "Gen2 OCR Pipeline")

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749667-f6315900-5f00-11eb-92bd-a297590adedc.png)](https://www.youtube.com/watch?v=YWIZYeixQjc "Gen2 OCR Pipeline")

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749638-eb76c400-5f00-11eb-8e9a-18e550b35ae4.png)](https://www.youtube.com/watch?v=Wclmk42Zvj4 "Gen2 OCR Pipeline")






