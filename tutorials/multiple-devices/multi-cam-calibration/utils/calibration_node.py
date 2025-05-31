import depthai as dai
from .img_annotation_helper import AnnotationHelper 
from typing import Optional, Tuple
import numpy as np
from types import SimpleNamespace
import cv2
import os

config = SimpleNamespace(
    checkerboard_size=(10, 7),  # number of squares on the checkerboard
    square_size=0.02,           # size of a square in meters
    calibration_data_dir="calibration_data"  # directory relative to main.py
)


class CalibrationNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.calib = False

        # Checkerboard properties
        self.checkerboard_size = config.checkerboard_size
        self.checkerboard_inner_size = (
            self.checkerboard_size[0] - 1,
            self.checkerboard_size[1] - 1,
        )
        self.square_size = config.square_size

        # Prepare 3D world coordinates for checkerboard corners
        self.corners_world = np.zeros(
            (1, self.checkerboard_inner_size[0] * self.checkerboard_inner_size[1], 3),
            np.float32,
        )
        self.corners_world[0, :, :2] = np.mgrid[
            0:self.checkerboard_inner_size[0], 0:self.checkerboard_inner_size[1]
        ].T.reshape(-1, 2)
        self.corners_world *= self.square_size

        # Initialize distortion coefficients (k1, k2, p1, p2, k3)
        self.distortion_coef = np.zeros((1, 5), dtype=np.float32)

        # Calibration results
        self.rot_vec: Optional[np.ndarray] = None
        self.trans_vec: Optional[np.ndarray] = None
        self.world_to_cam: Optional[np.ndarray] = None
        self.cam_to_world: Optional[np.ndarray] = None

        # Intrinsic matrices and device info
        self.intrinsic_mat_still: Optional[np.ndarray] = None # For high-res still image
        self.intrinsic_mat_preview: Optional[np.ndarray] = None # For preview image (drawing)
        self.device: Optional[dai.Device] = None
        self.mxid: Optional[str] = None
        self.preview_width: int = 0
        self.preview_height: int = 0
        self.still_width: int = 3840
        self.still_height: int = 2160

        # State for calibration process
        self.captured_still_for_calibration: Optional[np.ndarray] = None
        self.perform_calibration_flag = False
        
        self.annotation_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

    def _create_dai_color(
        self, r: float, g: float, b: float, a: float = 1.0
    ) -> dai.Color:
        color = dai.Color()
        color.r = r
        color.g = g
        color.b = b
        color.a = a
        return color

    def build(
        self,
        preview_input: dai.Node.Output,
        intrinsic_mat_still: np.ndarray, # Intrinsics for the full-resolution still
        intrinsic_mat_preview: np.ndarray,# Intrinsics for the preview resolution
        device: dai.Device,
        preview_width: int,
        preview_height: int,
        still_width: int = 3840,
        still_height: int = 2160,
    ) -> "CalibrationNode":
        self.intrinsic_mat_still = intrinsic_mat_still
        self.intrinsic_mat_preview = intrinsic_mat_preview
        self.device = device
        self.mxid = device.getDeviceId()
        self.preview_width = preview_width
        self.preview_height = preview_height
        self.still_width = still_width
        self.still_height = still_height

        self.link_args(preview_input)
        return self

    def trigger_calibration_with_image(self, image_bgr: np.ndarray):
        """
        Called by the main application thread when a still image has been captured
        and is ready for the calibration process.
        """
        if not self.calib:
            return
        self.captured_still_for_calibration = image_bgr
        self.perform_calibration_flag = True 

    def _estimate_pose_from_image(self, frame_bgr: np.ndarray) -> bool:
        """
        Performs checkerboard detection and pose estimation (solvePnP)
        on the provided BGR image. Updates calibration parameters if successful.
        """
        print(f"[{self.mxid}] INFO: Starting pose estimation from image...")
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        print(f"[{self.mxid}] INFO: Finding checkerboard corners...")
        found, corners = cv2.findChessboardCorners(
            frame_gray,
            self.checkerboard_inner_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH    # Use adaptive thresholding
            + cv2.CALIB_CB_FAST_CHECK       # Speed up by rejecting images without corners
            + cv2.CALIB_CB_NORMALIZE_IMAGE, # Normalize image brightness
        )

        if not found:
            print(f"[{self.mxid}] WARNING: Checkerboard not found in the image.")
            return False
        
        print(f"[{self.mxid}] INFO: Checkerboard found. Refining corner locations...")
        # Refine corner locations for better accuracy
        corners_refined = cv2.cornerSubPix(
            frame_gray,
            corners,
            (11, 11), # winSize: Half of the side length of the search window.
            (-1, -1), # zeroZone: Half of the size of the dead region in the middle of the search zone.
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), # criteria
        )

        print(f"[{self.mxid}] INFO: Computing camera pose using solvePnP...")
        # Compute rotation and translation vectors (object pose in camera coordinates)
        # Use the intrinsic matrix and distortion coefficients for the *still image* resolution
        ret, rvec, tvec = cv2.solvePnP(
            self.corners_world,         # 3D points in world coordinates
            corners_refined,            # Corresponding 2D points in image
            self.intrinsic_mat_still,   # Camera intrinsic matrix (for high-res still)
            self.distortion_coef,       # Distortion coefficients (k1,k2,p1,p2,k3)
        )

        if not ret:
            print(f"[{self.mxid}] ERROR: solvePnP failed to compute pose.")
            return False

        self.rot_vec = rvec
        self.trans_vec = tvec

        # Calculate transformation matrices
        rotM, _ = cv2.Rodrigues(self.rot_vec) # Convert rotation vector to rotation matrix
        self.world_to_cam = np.vstack(
            (np.hstack((rotM, self.trans_vec.reshape(3,1))), np.array([0, 0, 0, 1]))
        )
        self.cam_to_world = np.linalg.inv(self.world_to_cam)

        print(f"[{self.mxid}] INFO: Pose estimated successfully.")
        print(f"  Rotation Vector (rvec):\n{self.rot_vec}")
        print(f"  Translation Vector (tvec):\n{self.trans_vec}")

        try:
            # Output path relative to the current working directory (where main.py is run)
            output_dir = os.path.join(os.getcwd(), config.calibration_data_dir)
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"extrinsics_{self.mxid}.npz")
            
            np.savez(
                file_path,
                world_to_cam=self.world_to_cam,
                cam_to_world=self.cam_to_world,
                rot_vec=self.rot_vec, # Original rvec
                trans_vec=self.trans_vec, # Original tvec
                intrinsic_matrix_still=self.intrinsic_mat_still, # Intrinsics used for this calibration
                distortion_coefficients=self.distortion_coef, # Distortion coefs used
                checkerboard_inner_size=self.checkerboard_inner_size,
                checkerboard_square_size=self.square_size
            )
            print(f"[{self.mxid}] INFO: Calibration data saved to {file_path}")
            return True
        except Exception as e:
            print(f"[{self.mxid}] ERROR: Could not save calibration data: {e}")
            return False

    def process(
        self, frame_message: dai.ImgFrame
    ) -> None:
        # --- Handle Calibration Request ---
        if self.perform_calibration_flag:
            if self.captured_still_for_calibration is not None:
                print(f"[{self.mxid}] INFO: CalibrationNode processing calibration request...")
                success = self._estimate_pose_from_image(self.captured_still_for_calibration)
                if success:
                    print(f"[{self.mxid}] INFO: CalibrationNode: Calibration Succeeded. Origin will be drawn if selected.")
                else:
                    print(f"[{self.mxid}] WARNING: CalibrationNode: Calibration Failed.")
                self.captured_still_for_calibration = None # Clear the image memory
            else:
                print(f"[{self.mxid}] WARNING: CalibrationNode flagged for calibration but no image was provided.")
            self.perform_calibration_flag = False # Reset the flag

        annotations_builder = AnnotationHelper()

        # Display calibration selection status
        status_text = f"Selected for Calib: {self.calib}"
        status_color = (0,0,0,1) if self.calib else (0.5,0.5,0.5,1)
        status_bg_color = (1,1,0,0.7) if self.calib else (0.8,0.8,0.8,0.7) # Yellow if selected
        annotations_builder.draw_text(
            text=status_text, position=(0.02, 0.05), 
            color=status_color, background_color=status_bg_color, size=10,
        )

        if self.calib:
             annotations_builder.draw_text(
                text="Press 'c' on visualizer to CAPTURE & CALIBRATE",
                position=(0.02, 0.1), color=(0,0,0,1), background_color=(1,1,1,0.7), size=10,
            )

        if self.calib and self.rot_vec is not None and self.trans_vec is not None and \
           self.intrinsic_mat_preview is not None and self.preview_width > 0 and self.preview_height > 0:
            
            axis_length = self.square_size * 2.5 # Length of axes in meters (e.g., 2.5 squares long)
            world_points_axes = np.float32([
                [0, 0, 0],             # Origin
                [axis_length, 0, 0],   # X-axis end
                [0, axis_length, 0],   # Y-axis end
                [0, 0, -axis_length]    # Z-axis end 
            ])

            img_points_axes, _ = cv2.projectPoints(
                world_points_axes,
                self.rot_vec,
                self.trans_vec,
                self.intrinsic_mat_still, 
                self.distortion_coef 
            )


            scale_x = self.preview_width / self.still_width
            scale_y = self.preview_height / self.still_height 

            scaled_projected_points_for_preview = []
            for pt_hr in img_points_axes.reshape(-1, 2):
                scaled_projected_points_for_preview.append([pt_hr[0] * scale_x, pt_hr[1] * scale_y])
            
            scaled_points_arr = np.array(scaled_projected_points_for_preview)

            p_origin, p_x, p_y, p_z = scaled_points_arr # These are now effectively in preview pixel coordinates

            # Helper to convert pixel coordinates to normalized (0-1) for AnnotationHelper
            def to_norm_coords(px_pt: np.ndarray) -> Tuple[float, float]:
                norm_x = np.clip(px_pt[0], 0, self.preview_width -1) / self.preview_width
                norm_y = np.clip(px_pt[1], 0, self.preview_height -1) / self.preview_height
                return (float(norm_x), float(norm_y))

            try:
                norm_origin = to_norm_coords(p_origin)
                norm_px = to_norm_coords(p_x)
                norm_py = to_norm_coords(p_y)
                norm_pz = to_norm_coords(p_z)

                # Draw lines for axes
                annotations_builder.draw_line(norm_origin, norm_px, color=(1,0,0,1), thickness=2) # X = Red
                annotations_builder.draw_line(norm_origin, norm_py, color=(0,1,0,1), thickness=2) # Y = Green
                annotations_builder.draw_line(norm_origin, norm_pz, color=(0,0,1,1), thickness=2) # Z = Blue
            except Exception as e:
                print(f"[{self.mxid}] ERROR: Failed to draw origin axes on preview: {e}")

        dai_annotations = annotations_builder.build(
            frame_message.getTimestampDevice(), 
            frame_message.getSequenceNum()
        )
        self.annotation_out.send(dai_annotations)
