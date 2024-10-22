from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket, PointcloudPacket
from ground_plane_estimator import GroundPlaneEstimator
import depthai_viewer as viewer
import numpy as np
import cv2
import subprocess
import select
import sys
import time
import depthai as dai

FPS = 10.0
ground = GroundPlaneEstimator()

with OakCamera() as oak:
    try:
        subprocess.Popen([sys.executable, "-m", "depthai_viewer", "--viewer-mode"], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass

    viewer.init("Depthai Viewer")
    viewer.connect()

    color = oak.camera('color')

    stereo = oak.create_stereo()
    stereo.config_stereo(align=color, lr_check=True, extended=False, subpixel=True, median=dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    
    """     config= stereo.node.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 2
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = 15000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.node.initialConfig.set(config) """

    pointcloud = oak.create_pointcloud(depth_input=stereo)

    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    q = oak.queue([
        pointcloud.out.main.set_name('pcl'),
        color.out.main.set_name('color'),
        stereo.out.depth.set_name('stereo'),
        imu.out.main.set_name('imu')
    ]).configure_syncing(enable_sync=True, threshold_ms=500//FPS).get_queue()
    # oak.show_graph()
    oak.start()

    # For camera to settle (necessary for stereo?)
    time.sleep(5)

    viewer.log_rigid3(f"Right", child_from_parent=([0, 0, 0], [0, 0, 0, 0]), xyz="RDF", timeless=True)
    viewer.log_rigid3(f"Cropped", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF", timeless=True)

    def draw_mesh():
        pos,ind,norm = ground.get_plane_mesh(size=500)
        viewer.log_mesh("Right/Plane", pos, indices=ind, normals=norm, albedo_factor=[0.5,1,0], timeless=False)

    while oak.running():

        packets = q.get()

        cvFrame = packets["color"].frame[..., ::-1] # BGR to RGB
        depth = packets["stereo"].frame
        pcl_packet: PointcloudPacket = packets["pcl"]
        points = pcl_packet.points
        imu_packet = packets["imu"]

        # Get gravity vector from IMU
        accel_data = imu_packet.acceleroMeter 
        gravity_vect = np.array([accel_data.x, accel_data.y, accel_data.z])

        colors_720 = cvFrame.reshape(-1, 3)
        viewer.log_points("Right/PointCloud", points.reshape(-1, 3), colors=colors_720)
        viewer.log_depth_image("depth/frame", depth, meter=1e3)
        viewer.log_image("video/color", cvFrame)

        _, plane_pc, objects_pc, success = ground.estimate_ground_plane(points, gravity_vect)
        
        if success == False:
            print('No ground detected')
            continue
        
        draw_mesh()
        viewer.log_points("Cropped/Plane_PCL", plane_pc)
        viewer.log_points("Cropped/Objects_PCL", objects_pc, colors=(0.2,1.0,0.6))
