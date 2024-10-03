from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket, PointcloudPacket
from depthai_sdk.classes.box_estimator import BoxEstimator
import depthai_viewer as viewer
import cv2
import subprocess
import select
import sys

FPS = 10.0
box = BoxEstimator(median_window=10)

with OakCamera() as oak:
    try:
        subprocess.Popen([sys.executable, "-m", "depthai_viewer", "--viewer-mode"], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass
    viewer.init("Depthai Viewer")
    viewer.connect()

    color = oak.create_camera('cam_c', resolution='800p', fps=20)

    model_config = {
        'source': 'roboflow', # Specify that we are downloading the model from Roboflow
        'model':'cardboard-box-u35qd/1',
        'key':'dDOP8nChA9rZUWUTG8ia' # Fake API key, replace with your own!
    }
    nn = oak.create_nn(model_config, color)
    nn.config_nn(conf_threshold=0.85)

    tof = oak.create_tof(fps=20)
    tof.set_align_to(color, output_size=(640, 400))
    tof.configure_tof(phaseShuffleTemporalFilter=True,
                      phaseUnwrappingLevel=2,
                      phaseUnwrapErrorThreshold=100)

    pointcloud = oak.create_pointcloud(tof)

    q = oak.queue([
        pointcloud.out.main.set_name('pcl'),
        tof.out.main.set_name('tof'),
        nn.out.main.set_name('nn'),
    ]).configure_syncing(enable_sync=True, threshold_ms=500//FPS).get_queue()
    # oak.show_graph()
    oak.start()

    viewer.log_rigid3(f"Right", child_from_parent=([0, 0, 0], [0, 0, 0, 0]), xyz="RDF", timeless=True)
    viewer.log_rigid3(f"Cropped", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF", timeless=True)

    def draw_mesh():
        pos,ind,norm = box.get_plane_mesh(size=500)
        viewer.log_mesh("Right/Plane", pos, indices=ind, normals=norm, albedo_factor=[0.5,1,0], timeless=True)

    if box.is_calibrated():
        draw_mesh()
    else:
        print("Calibrate first, write 'c' in terminal when most of the view is flat floor!!")

    while oak.running():
        packets = q.get()

        nn: DetectionPacket = packets["nn"]
        cvFrame = nn.frame[..., ::-1] # BGR to RGB
        depth = packets["tof"].frame
        pcl_packet: PointcloudPacket = packets["pcl"]
        points = pcl_packet.points

        # Convert 800P into 400P into 256000x3
        colors_640 = cv2.pyrDown(cvFrame).reshape(-1, 3)
        viewer.log_points("Right/PointCloud", points.reshape(-1, 3), colors=colors_640)
        # Depth map visualize
        viewer.log_depth_image("depth/frame", depth, meter=1e3)
        viewer.log_image("video/color", cvFrame)

        if box.is_calibrated():
            if 0 == len(nn.detections):
                continue # No boxes found
            # Currently supports only 1 detection (box) at a time
            det = nn.detections[0]
            # Get the bounding box of the detection (relative to full frame size)
            # Add 10% padding on all sides
            box_bb = nn.bbox.get_relative_bbox(det.bbox)
            padded_box_bb = box_bb.add_padding(0.1)
            points_roi = pcl_packet.crop_points(padded_box_bb).reshape(-1, 3)

            dimensions, corners = box.process_points(points_roi)
            if corners is None:
                continue

            viewer.log_points("Cropped/Box_PCL", box.box_pcl)
            viewer.log_points("Cropped/Plane_PCL", box.plane_pcl, colors=(0.2,1.0,0.6))
            # viewer.log_points("Cropped/TopSide_PCL", box.top_side_pcl, colors=(1,0.3,0.6))
            viewer.log_points(f"Cropped/Box_Corners", corners, radii=8, colors=(1.0,0,0.0))

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

        key = oak.poll()
        ready, _, _ = select.select([sys.stdin], [], [], 0.001) # Terminal input
        if ready:
            key = sys.stdin.readline().strip()
        if key == 'c':
            if box.calibrate(points):
                print(f"Calibrated Plane: {box.ground_plane_eq}")
                draw_mesh()
