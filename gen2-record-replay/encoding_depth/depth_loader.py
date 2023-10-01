from pathlib import Path
import numpy as np
import cv2
import json
import os


class DepthFrame:
    def __init__(self, depth: np.ndarray, json: dict) -> None:
        # Check metadata validity
        if json is None:
            raise ValueError("Metadata json cannot be None")
        if "baseline" not in json:
            raise ValueError("Metadata must contain baseline")
        self.baseline = json["baseline"]
        if "focal_length" not in json:
            raise ValueError("Metadata must contain focal_length")
        self.focal_length = json["focal_length"]
        if "max_disparity" not in json:
            raise ValueError("Metadata must contain max_disparity")
        self.max_disparity = json["max_disparity"]
        if "disp_to_depth_factor" not in json:
            raise ValueError("Metadata must contain disp_to_depth_factor")
        self.disp_to_depth_factor = json["disp_to_depth_factor"]

        if "num_subpixels" not in json:
            raise ValueError("Metadata must contain num_subpixels")
        self.num_subpixels = json["num_subpixels"]

        # Check depth validity
        if depth is None:
            raise ValueError("Depth cannot be None")
        if depth.dtype != np.uint16:
            raise ValueError("Depth must be uint16")
        if depth.ndim != 2:
            raise ValueError("Depth must be 2-dimensional")
        self.depth_frame = depth

    def get_depth(self) -> np.ndarray:
        return self.depth_frame

    def get_disparity(self) -> np.ndarray:
        # Ignore division by zero warning
        np.seterr(divide='ignore', invalid='ignore')
        disp = self.disp_to_depth_factor * self.num_subpixels / self.depth_frame
        # On the cases where the depth is 0, the disparity is also 0
        disp = np.where(self.depth_frame == 0, 0, disp)
        return np.round(disp).astype(np.uint16)

    def get_preview_disparity(self) -> cv2.Mat:
        disparity = self.get_disparity()
        disparity = (disparity * (255 / (self.max_disparity * self.num_subpixels))).astype(np.uint8)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
        return disparity


class DepthLoader:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        self.current_index = 0
        with open(str(self.dataset_path / "depth_info.json"), "r") as f:
            self.json_info = json.load(f)
        self.depth_frames : np.ndarray = np.load(str(self.dataset_path / "depth_frames.npy"))
        # Check that the depth frames are valid
        if self.depth_frames.dtype != np.uint16:
            raise ValueError("Depth frames must be uint16")
        if self.depth_frames.ndim != 3:
            raise ValueError("Depth frames must be 3-dimensional")
        if self.depth_frames.shape[0] != self.json_info["num_frames"]:
            raise ValueError(
                f"Depth frames must have {self.json_info['num_frames']} frames, but has {self.depth_frames.shape[0]}"
            )

    # Return the next frame, unless there are no more frames, then return None
    def get_next_frame(self) -> DepthFrame:
        depth_frame_np = self.depth_frames[self.current_index, :, :]
        self.current_index += 1
        if self.current_index >= self.depth_frames.shape[0]:
            return None
        else:
            return DepthFrame(depth_frame_np, self.json_info)


if __name__ == "__main__":
    # Get the testDataset from the parent directory
    loader = DepthLoader(Path(__file__).parent / "testData")
    while True:
        frame = loader.get_next_frame()
        if frame is None:
            print("Reached the end of the depth frames")
            break
        cv2.imshow("colorized disparity", frame.get_preview_disparity())
        if cv2.waitKey(100) == ord('q'):
            break
