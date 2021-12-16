# Calculate spatial coordinates on the host

This example shows how to calcualte spatial coordinates of a ROI on the host and gets depth frames from the device. Other option is to use [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/components/nodes/spatial_location_calculator/) to calcualte spatial coordinates on the device.

If you already have depth frames and ROI (Region-Of-Interest, eg. bounding box of an object) / POI (Point-Of-Interest, eg. feature/key-point) on
the host, it might be easier to just calculate the spatial coordiantes of that region/point on the host, instead of sending depth/ROI back
to the device.

**Note** using single points / tiny ROIs (eg. 3x3) should be avoided, as depth frame can be noisy, so you should use **at least 10x10 depth pixels
ROI**. Also note that to maximize spatial coordinates accuracy, you should define min and max threshold accurately.

## Demo

![Demo](https://user-images.githubusercontent.com/18037362/146296930-9e7071f5-33b9-45f9-af21-cace7ffffc0f.gif)

## Installation

```
python3 -m pip install -r requirements.txt
```
