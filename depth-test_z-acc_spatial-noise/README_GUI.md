# Insturctions to run the GUI for RVC3 depth test

## How to install
 * Follow instructions to get started for your platform here: https://docs.luxonis.com/projects/api/en/latest/install/#installation
 * Install the requirements for this test:
   * `pip3 install -r requirements.txt`

## How to run
* Run the GUI:
    * `python3 gui.py --resultsPath <path to save results>` where `<path to save results>` is the path where you want to save the results
    * check `gui.py --help` for more options
* Select the device ID
* Select the distance and edge
* Perform the measurement in the following way:
    * Move the camera to the starting position
    * Click Record frames (press `q` to stop recording)
    * Click Frames to depth, to transform the frames to depth map
    * Click measure to measure the distance from the ROI (the script will run twice, once for vertical and once for horizontal)
        * Click on the gray image and select the ROI
        * Click on the pointcloud, press `f` to fit a plane and `t` to measure the distance
        * The distance is printed out in the console where GUI was started
        * Press `q` to exit, while having the focus on the pointcloud window
        


