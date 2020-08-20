# CoronaMask

This experiment allows you to run the COVID-19 mask/no-mask object detector which was trained via the Google Colab tutorial [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-).

Below is a quick run of this mask detector:

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/90733159-74436100-e2cc-11ea-8fb6-d4be937d90e5.gif)](https://photos.app.goo.gl/mJZ8TdWoNatHzW4x7 "COVID-19 mask detection")

You can leverage this notebook ([here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-)) and add your own training images to improve the quality of this mask detector.  Our training was a quick weekend effort at it.  

## Install

```
python3 -m pip install -r requirements.txt
```

## Run

```
python3 main.py
```
