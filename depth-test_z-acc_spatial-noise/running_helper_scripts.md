# Instructions for running helper scripts for depth testing in batches
Note: This is a work in progress. The scripts are not well generalized and may require some manual editing to run.



* All scripts expect the following directory structure:
```
DepthTestExample
├── camera_15
│   ├── 10.0m
│   │   └── center
│   │       └── 1-53F2B816838860C6
│   │           ├── camb,c.avi
│   │           ├── camc,c.avi
│   │           ├── camd,c.avi
├── camera_17
    ├── 10.0m
    │   └── center
        |   └── 1-3B929016838860C6
        |       ├── camb,c.avi
        |       ├── camc,c.avi
        |       ├── camd,c.avi
        └── left
            └── 1-3B929016838860C6
                ├── camb,c.avi
                ├── camc,c.avi
                ├── camd,c.avi
```


## How to acquire depth testing videos

* Depth measurements can be acquired using any tool that can produce the directory structure that's shown above, at the moment - this is possible with two tool.
* One is `gui.py` from this repository.
* The other is https://github.com/moratom/rvc3_recorder - instructions on how to run it in the README.md file of that repository. The rvc3_recorder is much faster to work with when many cameras need to be captured, because the videos are run continuously.


## Labeling the data for depth testing

* Run `python measure_all.py ~/DepthTestExample --camera_ids 15 17 -mode roi_selection --positions center` to label the data for depth testing. This will create a file roi.txt in each of the directories. See `--help` for more options. After each image is shown - select the ROI that you want to depth test and press `q`


## Running the depth testing
* When data is labeled in advance depth testing can be run autonomously for any given number of calibrations for each camera.
* For each camera - create a directory `calibrations` inside the the camera directory as such:
```
DepthTestExample
├── camera_15
│   ├── 10.0m
│   │   └── center
│   │       └── 1-53F2B816838860C6
│   │           ├── camb,c.avi
│   │           ├── camc,c.avi
│   │           ├── camd,c.avi
│   │           └── roi.txt
|   |──calibrations
|   |   |──calibration_1.json
|   |   |──calibration_2.json
```
* Then run:
```
python measure_all.py ~/DepthTestExample --camera_ids 15 17 -mode measure --alpha 1
```
to run depth testing for each camera. This will run the visualiations as well - this is useful to make sure everything is running as expected. If you want to run thing in the background use:
```
Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1; python measure_all.py ~/DepthTestExample --camera_ids 15 17 -mode measure --alpha 1
```

* This will create a `depthOCV` directory inside every test directory and inside create `calibration_1` and `calibration_2` each with it's own results.

* To merge all the depth tests inside a single .csv file run:
```
python merge_all.py --input_dir ~/DepthTestExample --camera_ids 15 17  --output_file ~/DepthTestExample/results.csv
```