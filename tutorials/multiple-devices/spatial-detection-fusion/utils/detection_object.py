import numpy as np

class WorldDetection:
    """
    Represents a detected object's state in the world coordinate system,
    derived from a tracklet.
    """
    def __init__(self, 
                 tracklet_id: int, 
                 label_str: str, 
                 label_idx: int,
                 pos_world_homogeneous: np.ndarray, # 4x1: [x, y, z, 1].T
                 camera_friendly_id: int,
                 confidence: float = 0.0 # Tracklets might not always carry original confidence
                ):
        self.tracklet_id: int = tracklet_id
        self.label_str: str = label_str
        self.label_idx: int = label_idx # Original label index from NN
        self.pos_world_homogeneous: np.ndarray = pos_world_homogeneous
        self.pos_world_cartesian: np.ndarray = pos_world_homogeneous[:3, 0] # Extract [x, y, z]
        self.camera_friendly_id: int = camera_friendly_id # Which camera reported this
        self.confidence: float = confidence # Store if available

        # For grouping logic, similar to your old Detection class
        self.corresponding_world_detections: list['WorldDetection'] = []

    def __repr__(self):
        return (f"WorldDetection(id={self.tracklet_id}, lbl='{self.label_str}', "
                f"pos_w=({self.pos_world_cartesian[0]:.2f}, {self.pos_world_cartesian[1]:.2f}, {self.pos_world_cartesian[2]:.2f}), "
                f"cam_id={self.camera_friendly_id})")

