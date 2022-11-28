[中文文档](README.zh-CN.md)

## [Gen2] Text Detection + Optical Character Recognition (OCR) Pipeline

This pipeline implements text detection (EAST) followed by optical character recognition of the detected text.  It implements issue [#124](https://github.com/luxonis/depthai/issues/124).

## How to Run

This example uses the [Gen2 Pipeline Builder](https://github.com/luxonis/depthai/issues/136), so make sure to install the requirements below to make sure the appropriate Gen2 API version.

`python3 -m pip install -r requirements.txt`

After installing the requirements, running this pipeline is as simple as:

`python3 main.py`

## Example Results

Upon running, you can point this at text of interest to get the detection and the resultant text in the detected areas (and the locations of the text in pixel space).  

Note that the more text in the frame, the slower the network will run - as it is running OCR on every detected region of text.  
And you will see this variance in the examples below:


[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749743-13febe00-5f01-11eb-8b5f-dca801f5d125.png)](https://www.youtube.com/watch?v=Bv-p76A3YMk "Gen2 OCR Pipeline")

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749667-f6315900-5f00-11eb-92bd-a297590adedc.png)](https://www.youtube.com/watch?v=YWIZYeixQjc "Gen2 OCR Pipeline")

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749638-eb76c400-5f00-11eb-8e9a-18e550b35ae4.png)](https://www.youtube.com/watch?v=Wclmk42Zvj4 "Gen2 OCR Pipeline")






