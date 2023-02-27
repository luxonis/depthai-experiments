[中文文档](README.zh-CN.md)

##  Text Detection + Optical Character Recognition (OCR) Pipeline

This pipeline implements text detection (EAST) followed by optical character recognition of the detected text.  It implements issue [#124](https://github.com/luxonis/depthai/issues/124).

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

## Example Results

Upon running, you can point this at text of interest to get the detection and the resultant text in the detected areas (and the locations of the text in pixel space).  

Note that the more text in the frame, the slower the network will run - as it is running OCR on every detected region of text.  
And you will see this variance in the examples below:


[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749743-13febe00-5f01-11eb-8b5f-dca801f5d125.png)](https://www.youtube.com/watch?v=Bv-p76A3YMk "Gen2 OCR Pipeline")

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749667-f6315900-5f00-11eb-92bd-a297590adedc.png)](https://www.youtube.com/watch?v=YWIZYeixQjc "Gen2 OCR Pipeline")

[![Text Detection + OCR on DepthAI](https://user-images.githubusercontent.com/32992551/105749638-eb76c400-5f00-11eb-8e9a-18e550b35ae4.png)](https://www.youtube.com/watch?v=Wclmk42Zvj4 "Gen2 OCR Pipeline")






