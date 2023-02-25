# DepthAI Pointcloud generation

We have two different pointcloud demos which generate the pointcloud either on the OAK device or on the host computer (using `open3d`).

### On-device Pointcloud NN model

[On-device Pointcloud NN model](device-pointcloud) - Uses NN model to generate pointcloud from the depth on the OAK camera itself.

![image](https://user-images.githubusercontent.com/18037362/158055419-5c80d524-3478-49e0-b7b8-099b07dd57fa.png)

### RGB-D projection

[RGB-D projection](rgbd-pointcloud) - Uses [open3d](http://www.open3d.org/)'s `PointCloud.create_from_depth_image()` function to calculate pointcloud from the depth map on the host. It also supports colorized pointcloud.

![img](https://user-images.githubusercontent.com/18037362/158277114-f1676487-e214-4872-a1b3-aa14131b666b.png)

### Multi-device pointcloud

See [demo here](https://github.com/luxonis/depthai-experiments/tree/master/multiple-devices/rgbd-pointcloud-fusion) for **pointcloud fusion from multiple OAKs**.

![demo](https://user-images.githubusercontent.com/18037362/198794141-be39c3c1-b6e8-4a8a-8b1b-e9c4e30e7365.gif)

