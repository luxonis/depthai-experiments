from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket, PointcloudPacket
from box_estimator_single import BoxEstimator
from ground_plane_estimator import GroundPlaneEstimator
import depthai_viewer as viewer
import numpy as np
import cv2
import subprocess
import select
import sys
import time
import open3d as o3d
from depthai_sdk.visualize.bbox import BoundingBox
import depthai as dai

"Idea1: dynamic ground plane estimation from the input point cloud. Everything else stays the same. + frame skipping"


FPS = 10.0
box = BoxEstimator(median_window=10)

ground = GroundPlaneEstimator()
bb = BoundingBox()

with OakCamera() as oak:
    try:
        subprocess.Popen([sys.executable, "-m", "depthai_viewer", "--viewer-mode"], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass
    viewer.init("Depthai Viewer")
    viewer.connect()

    color = oak.camera('color')
    
    model_config = {
        'source': 'roboflow', # Specify that we are downloading the model from Roboflow
        'model':'cardboard-box-u35qd/1',
        'key':'pPZRmrmqoNcQNt6oluzh' # Fake API key, replace with your own!
    }

    nn = oak.create_nn(model_config, color)
    nn.config_nn(conf_threshold=0.85)

    stereo = oak.create_stereo()
    stereo.config_stereo(align=color, lr_check=True, extended=False, subpixel=True, median=dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    
    """     config= stereo.node.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = 15000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.node.initialConfig.set(config) """

    pointcloud = oak.create_pointcloud(depth_input=stereo)

    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    q = oak.queue([
        pointcloud.out.main.set_name('pcl'),
        stereo.out.depth.set_name('stereo'),
        nn.out.main.set_name('nn'),
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

    # For frame skipping 
    frame_counter = 0
    frame_skip = 2      # Process every 2nd frame

    while oak.running():
        packets = q.get()

        nn: DetectionPacket = packets["nn"]
        cvFrame = nn.frame[..., ::-1] # BGR to RGB
        depth = packets["stereo"].frame
        pcl_packet: PointcloudPacket = packets["pcl"]
        points = pcl_packet.points

        imu_packet = packets["imu"]
        accel_data = imu_packet.acceleroMeter  # Acceleration data

        gravity_vect = np.array([accel_data.x, accel_data.y, accel_data.z])

        colors_720 = cvFrame.reshape(-1, 3)        
        viewer.log_points("Right/PointCloud", points.reshape(-1, 3), colors=colors_720)
        viewer.log_depth_image("depth/frame", depth, meter=1e3)
        viewer.log_image("video/color", cvFrame)

        if frame_counter % frame_skip == 0:

            ground_plane_eq, ground_plane_pc, objects_pc, success = ground.estimate_ground_plane(points, gravity_vect)
            if success == False:
                print('No ground detected')
                continue

            draw_mesh()

            if 0 == len(nn.detections):
                continue # No boxes found

            # Only 1 detection (box) at a time
            det = nn.detections[0]
            # Get the bounding box of the detection (relative to full frame size)
            # Add 10% padding on all sides
            box_bb = nn.bbox.get_relative_bbox(det.bbox)
            padded_box_bb = box_bb.add_padding(0.1)
            points_roi = pcl_packet.crop_points(padded_box_bb).reshape(-1, 3)
            
            dimensions, corners = box.process_points(ground_plane_eq, points_roi)

            if corners is None:
                    continue

            viewer.log_points("Cropped/Box_PCL", box.box_pcl)
            viewer.log_points("Cropped/Plane_PCL", box.plane_pcl, colors=(0.2,1.0,0.6))
            viewer.log_points(f"Cropped/Box_Corners", corners, radii=8, colors=(1.0,0,0.0))
            viewer.log_line_segments(f"Cropped/Box_Edges", box.get_3d_lines(corners), stroke_width=4, color=(1.0,0,0.0))

            corners = box.inverse_corner_points()
            viewer.log_points(f"Right/Box_Corners", corners, radii=8, colors=(1.0,0,0.0))
            viewer.log_line_segments(f"Right/Box_Edges", box.get_3d_lines(corners), stroke_width=4, color=(1.0,0,0.0))

            l,w,h = dimensions
            label = f"{det.label_str} ({det.confidence:.2f})\n{l/10:.1f} x {w/10:.1f}\nH: {h/10:.1f} cm"
            
            viewer.log_rect('video/bbs',
                    box_bb.to_tuple(cvFrame.shape),
                    label=label,
                    rect_format=viewer.RectFormat.XYXY)
            viewer.log_rect('depth/bbs',
                    padded_box_bb.to_tuple(depth.shape),
                    label="Padded BoundingBox",
                    rect_format=viewer.RectFormat.XYXY)
                
        frame_counter += 1
