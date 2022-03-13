# DepthAI pointcloud generation

We have two different pointcloud demos:

- [On-device Pointcloud NN model](device-pointcloud) - Uses NN model to generate pointcloud from the depth on the OAK camera itself.
- [RGB-D projection](rgbd-pointcloud) - Uses [open3d](http://www.open3d.org/)'s `PointCloud.create_from_depth_image()` function to calculate pointcloud from the depth map on the host. It also supports colorized pointcloud.