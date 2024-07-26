import depthai as dai
import numpy as np
import rerun as rr
import cv2

class Rerun(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, color: dai.Node.Output, depth: dai.Node.Output, device: dai.Device) -> "Rerun":
        self.link_args(color, depth)
        self.sendProcessingToPipeline(True)
        self.device = device
        self.xyz = None
        return self

    def process(self, color: dai.ImgFrame, depth: dai.ImgFrame) -> None:
        depth_frame = depth.getCvFrame()

        if self.xyz is None:
            self.xyz = create_xyz(self.device, depth_frame.shape[1], depth_frame.shape[0])

        depth_frame = np.expand_dims(np.array(depth_frame), axis=-1)
        # To meters and reshape for rerun
        pcl = (self.xyz * depth_frame / 1000.0).reshape(-1, 3)
        colors = cv2.cvtColor(color.getCvFrame(), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        rr.log("Pointcloud", rr.Points3D(pcl, colors=colors))


def create_xyz(device, width, height):
    calibData = device.readCalibration()
    M_right = calibData.getCameraIntrinsics(
        dai.CameraBoardSocket.RIGHT, dai.Size2f(width, height))
    camera_matrix = np.array(M_right).reshape(3, 3)

    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
    points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

    # unpack coordinates
    u_coord: np.array = points_2d[..., 0]
    v_coord: np.array = points_2d[..., 1]

    # unpack intrinsics
    fx: np.array = camera_matrix[0, 0]
    fy: np.array = camera_matrix[1, 1]
    cx: np.array = camera_matrix[0, 2]
    cy: np.array = camera_matrix[1, 2]

    # projective
    x_coord: np.array = (u_coord - cx) / fx
    y_coord: np.array = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord], axis=-1)
    return np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
