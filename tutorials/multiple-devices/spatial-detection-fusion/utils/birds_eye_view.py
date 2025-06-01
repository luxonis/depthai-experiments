import depthai as dai
import numpy as np
import cv2
import collections
from typing import Dict, List, Any, Tuple, Optional

from utils.detection_object import WorldDetection


class BirdsEyeViewNode(dai.node.HostNode):
    colors = [ 
        (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), 
        (0, 255, 0), (255, 0, 0), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        (128,128,128), (255,255,255), (0,128,128), (128,128,0)
    ]

    def __init__(self): 
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)],
        )
        
        self.config = None
        self.all_cam_extrinsics: Optional[Dict[str, Dict[str, Any]]] = None
        self.width: int = 0
        self.height: int = 0
        self.scale: float = 0.0
        self.trail_length: int = 0
        self.label_map: Optional[List[str]] = None
        self.world_to_birds_eye: Optional[np.ndarray] = None
        self.bev_img_display: Optional[np.ndarray] = None
        self.history: Optional[collections.deque] = None
        
        # To store the latest processed detections from each camera
        self.latest_world_detections_per_cam: Dict[str, List[WorldDetection]] = {}


    def build(self, 
              all_cam_extrinsics: Dict[str, Dict[str, Any]], 
              node_config: Any,
              all_trackers_outputs: Dict[str, dai.Node.Output]) -> "BirdsEyeViewNode":
        self.config = node_config

        self.all_cam_extrinsics = all_cam_extrinsics
        self.width = self.config.bev_width
        self.height = self.config.bev_height
        self.scale = self.config.bev_scale 
        self.trail_length = self.config.trail_length
        self.label_map = self.config.label_map

        self.world_to_birds_eye = np.array([
            [self.scale, 0,   0, self.width // 2],
            [0, self.scale, 0, self.height // 2],
        ], dtype=np.float32)

        self.bev_img_display = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.history = collections.deque(maxlen=self.trail_length)

        for mxid in self.all_cam_extrinsics.keys():
            input_name = f"tracklets_{mxid}"
            self.inputs[input_name].setBlocking(False)
            self.inputs[input_name].setMaxSize(1) # Process latest tracklet message
            self.latest_world_detections_per_cam[mxid] = [] # Initialize cache
            print(f"    BEVNode: Created input port '{input_name}'")

        self.link_args(tracklet_out)
        return self

    def process(self, tracklets_msg: dai.Tracklets):
        mxid = msg_name.split("tracklets_")[1]

        if mxid not in self.all_cam_extrinsics:
            # This should not happen if inputs are created based on all_cam_extrinsics
            print(f"BEVNode: Received data for unknown MXID '{mxid}' from port '{msg_name}'. Skipping.")
            return

        extrinsics = self.all_cam_extrinsics[mxid]
        cam_to_world_matrix = extrinsics.get('cam_to_world')
        friendly_id = extrinsics.get('friendly_id', 0)

        if cam_to_world_matrix is None:
            self.latest_world_detections_per_cam[mxid] = [] # Clear data if extrinsics are missing
            self._trigger_render() # Re-render to reflect missing data
            return

        current_cam_world_detections: List[WorldDetection] = []
        for tracklet in tracklets_msg.tracklets:
            if np.isnan(tracklet.spatialCoordinates.x) or \
                np.isnan(tracklet.spatialCoordinates.y) or \
                np.isnan(tracklet.spatialCoordinates.z):
                continue 

            pos_cam_m = np.array([
                tracklet.spatialCoordinates.x / 1000.0,
                tracklet.spatialCoordinates.y / 1000.0, 
                tracklet.spatialCoordinates.z / 1000.0,
                1.0 
            ]).reshape(4, 1)

            pos_world_homogeneous = cam_to_world_matrix @ pos_cam_m

            label_idx = tracklet.label
            label_str = self.label_map[label_idx] if 0 <= label_idx < len(self.label_map) else f"L{label_idx}"

            confidence = 0.0
            if hasattr(tracklet, 'srcImgDetection') and tracklet.srcImgDetection is not None:
                confidence = getattr(tracklet.srcImgDetection, 'confidence', 0.0)

            world_det = WorldDetection(
                tracklet_id=tracklet.id, label_str=label_str, label_idx=label_idx,
                pos_world_homogeneous=pos_world_homogeneous,
                camera_friendly_id=friendly_id, confidence=confidence
            )
            current_cam_world_detections.append(world_det)

        self.latest_world_detections_per_cam[mxid] = current_cam_world_detections
        self._trigger_render()

    def _trigger_render(self):
        """Consolidates all current detections and renders the BEV."""
        if self.bev_img_display is None: return

        all_current_world_detections: List[WorldDetection] = []
        for det_list in self.latest_world_detections_per_cam.values():
            all_current_world_detections.extend(det_list)

        self.bev_img_display.fill(0) 
        self._draw_coordinate_system()
        self._draw_cameras()

        groups = self._make_groups(all_current_world_detections)

        self._draw_history() 
        if all_current_world_detections or groups:
            self.history.append(groups) 
        self._draw_groups(groups)   

        img_frame_out = dai.ImgFrame()
        img_frame_out.setType(dai.ImgFrame.Type.BGR888p) 
        img_frame_out.setWidth(self.width)
        img_frame_out.setHeight(self.height)
        img_frame_out.setData(self.bev_img_display.tobytes()) 
        self.output.send(img_frame_out)

    def _project_to_bev(self, pos_world_homogeneous: np.ndarray) -> Optional[Tuple[int, int]]:
        if self.world_to_birds_eye is None: return None
        if pos_world_homogeneous.shape[0] < 3: return None
        if pos_world_homogeneous.shape == (3,1) or pos_world_homogeneous.shape == (3,):
            pos_world_homogeneous = np.append(pos_world_homogeneous.flatten()[:3], 1.0).reshape(4,1)
        elif pos_world_homogeneous.shape != (4,1):
            pos_world_homogeneous = np.array(pos_world_homogeneous[:4]).reshape(4,1)
        bev_coords = self.world_to_birds_eye @ pos_world_homogeneous
        u, v = int(bev_coords[0,0]), int(bev_coords[1,0])
        if 0 <= u < self.width and 0 <= v < self.height: return u, v
        return None

    def _draw_coordinate_system(self):
        if self.bev_img_display is None: return
        origin_bev = self._project_to_bev(np.array([0,0,0,1], dtype=np.float32))
        x_axis_bev = self._project_to_bev(np.array([1.0,0,0,1], dtype=np.float32)) 
        y_axis_bev = self._project_to_bev(np.array([0,1.0,0,1], dtype=np.float32)) 
        if origin_bev and x_axis_bev: cv2.line(self.bev_img_display, origin_bev, x_axis_bev, (0,0,255),1)
        if origin_bev and y_axis_bev: cv2.line(self.bev_img_display, origin_bev, y_axis_bev, (0,255,0),1)

    def _draw_cameras(self):
        if self.bev_img_display is None or self.all_cam_extrinsics is None: return
        for mxid, extrinsics in self.all_cam_extrinsics.items():
            cam_to_world = extrinsics.get('cam_to_world')
            friendly_id = extrinsics.get('friendly_id', 0)
            if cam_to_world is None: continue
            color_idx = (friendly_id -1) % len(self.colors) 
            color = self.colors[color_idx]
            cam_origin_w = cam_to_world @ np.array([0,0,0,1], dtype=np.float32)
            cam_origin_bev = self._project_to_bev(cam_origin_w)
            if cam_origin_bev:
                cv2.circle(self.bev_img_display, cam_origin_bev, 6, color, -1)
                cv2.putText(self.bev_img_display, str(friendly_id), (cam_origin_bev[0]+10, cam_origin_bev[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cam_fwd_cam_space = np.array([0,0,0.5,1], dtype=np.float32) 
                cam_fwd_w = cam_to_world @ cam_fwd_cam_space
                cam_fwd_bev = self._project_to_bev(cam_fwd_w)
                if cam_fwd_bev: cv2.line(self.bev_img_display, cam_origin_bev, cam_fwd_bev, color, 1)

    def _make_groups(self, world_detections: List[WorldDetection]) -> List[List[WorldDetection]]:
        if not world_detections: return []
        n_components_for_dist = 2 
        distance_threshold = 0.8  
        for det in world_detections: det.corresponding_world_detections = []
        for i in range(len(world_detections)):
            for j in range(i + 1, len(world_detections)):
                det1 = world_detections[i]; det2 = world_detections[j]
                if det1.camera_friendly_id == det2.camera_friendly_id: continue
                if det1.label_str == det2.label_str:
                    dist = np.linalg.norm(det1.pos_world_cartesian[:n_components_for_dist] - det2.pos_world_cartesian[:n_components_for_dist])
                    if dist < distance_threshold:
                        det1.corresponding_world_detections.append(det2)
                        det2.corresponding_world_detections.append(det1)
        groups_final: List[List[WorldDetection]] = []
        processed_in_group = set() 
        for det_start_node in world_detections:
            if det_start_node in processed_in_group: continue
            current_group_set = set(); queue = collections.deque([det_start_node])
            processed_in_group.add(det_start_node); current_group_set.add(det_start_node)
            while queue:
                current_det = queue.popleft()
                for neighbor in current_det.corresponding_world_detections:
                    if neighbor not in processed_in_group:
                        processed_in_group.add(neighbor); current_group_set.add(neighbor); queue.append(neighbor)
            if current_group_set: groups_final.append(list(current_group_set))
        return groups_final

    def _draw_groups(self, groups: List[List[WorldDetection]]):
        if self.bev_img_display is None: return
        for group in groups:
            if not group: continue
            sum_pos_world_cartesian = np.zeros(3, dtype=np.float32)
            for det in group: sum_pos_world_cartesian += det.pos_world_cartesian
            avg_pos_world_cartesian = sum_pos_world_cartesian / len(group)
            avg_pos_world_hom = np.append(avg_pos_world_cartesian, 1.0)
            avg_pos_bev = self._project_to_bev(avg_pos_world_hom)
            label = group[0].label_str
            if avg_pos_bev:
                group_radius_px = int(0.12 * self.scale) 
                cv2.circle(self.bev_img_display, avg_pos_bev, group_radius_px, (220,220,220), 1)
                cv2.putText(self.bev_img_display, label, (avg_pos_bev[0]+group_radius_px+3, avg_pos_bev[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,220),1)
                for det in group:
                    det_pos_bev = self._project_to_bev(det.pos_world_homogeneous)
                    if det_pos_bev:
                        color_idx = (det.camera_friendly_id -1) % len(self.colors)
                        color = self.colors[color_idx]
                        cv2.circle(self.bev_img_display, det_pos_bev, 2, color, -1)

    def _draw_history(self):
        if self.bev_img_display is None or self.history is None: return
        for i, groups_in_hist_step in enumerate(self.history):
            for group in groups_in_hist_step:
                if not group: continue
                sum_pos_world_cartesian = np.zeros(3, dtype=np.float32)
                num_valid_dets_in_group = 0
                for det in group:
                    if det.pos_world_cartesian is not None:
                        sum_pos_world_cartesian += det.pos_world_cartesian
                        num_valid_dets_in_group +=1
                if num_valid_dets_in_group == 0: continue
                avg_pos_world_cartesian = sum_pos_world_cartesian / num_valid_dets_in_group
                avg_pos_world_hom = np.append(avg_pos_world_cartesian, 1.0)
                avg_pos_bev = self._project_to_bev(avg_pos_world_hom)
                if avg_pos_bev:
                    intensity = int((i / float(self.trail_length)) * 100) + 30 
                    intensity = min(max(intensity, 20), 150) 
                    radius = int((i / float(self.trail_length)) * 3) + 1 
                    cv2.circle(self.bev_img_display, avg_pos_bev, radius, (intensity,intensity,intensity), -1)

