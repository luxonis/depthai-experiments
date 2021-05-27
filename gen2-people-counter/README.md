# Gen2 People counting

This demo uses [person_detection_retail_0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html) neural network to detect people. The demo also displays the number of people detected on the frame.

By default, it will use images from `/images` folder and change the image every 3 seconds. You can also use color camera input (instead of images from the host) using `-cam` argument.

To use a different NN model (eg. `MobileNet SSD` or `pedestrian_detection_adas_0002`), you would have to change the `size` (input size of the NN) and `nnPath` variables.
