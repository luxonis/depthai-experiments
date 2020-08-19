# depthai-experiments
Experimental projects we've done with DepthAI.  

**Experiments** can be anything from "here's some code and it works sometimes" to "this is almost a tutorial".  

The following list isn't exhaustive (as we randomly add experiments and we may forget to update this list), but here are some as of June 1st 2020:

## COVID-19 Mask / No-Mask Detector ([here](https://github.com/luxonis/depthai-experiments/blob/master/coronamask/README.md))

This project shows you how to run the COVID-19 mask/no-mask object detector which was trained [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-)

[![COVID-19 mask-no-mask megaAI](https://i.imgur.com/iZMigOv.png)](https://photos.app.goo.gl/mJZ8TdWoNatHzW4x7 "COVID-19 mask detection")

## Social Distancing Example ([here](https://github.com/luxonis/depthai-experiments/tree/master/social-distancing))
Since DepthAI gives the full 3D position of objects in physical space, it's a couple lines of code to make a social-distancing monitor with DepthAI.  So that's what this project is, a quick cut at a social-distancing monitor.

[![COVID-19 Social Distancing with DepthAI](https://i.imgur.com/17WR2v9.jpg)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

## Demo-UI ([here](https://github.com/luxonis/depthai-experiments/tree/master/demo-ui))
Application used to demonstrate various capabilities of the DepthAI platform. Contains examples with descriptions,
console outputs and preview windows.

![DemoUI](./demo-ui/preview.png)

## MJPEG and JSON streaming ([here](https://github.com/luxonis/depthai-experiments/tree/master/mjpeg-streaming))

[![MJPEG Streaming DepthAI](https://i.imgur.com/0DT3NNR.jpg)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

This lay the groundwork to make DepthAI [OpenDataCam](https://github.com/opendatacam/opendatacam) compatible.

## Stereo Neural Inference Results Visualizer ([here](https://github.com/luxonis/depthai-experiments/tree/master/triangulation-3D-visualizer))

So because there are often application-specific host-side filtering to be done on the stereo neural inference results, and because these calculations are lightweight (i.e. could be done on an ESP32), we leave the triangulation itself to the host.  If there is interest to do this on DepthAI directly instead, please let us know!

[![Spatial AI](https://user-images.githubusercontent.com/32992551/89942141-44fc6800-dbd9-11ea-8142-fe126922148f.png)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")






