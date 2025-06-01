bev_width = 800  # Width of the Bird's-Eye View window in pixels
bev_height = 800 # Height of the Bird's-Eye View window in pixels
bev_scale = 40   # Pixels per meter for the Bird's-Eye View (e.g., 1 meter = 40 pixels)
trail_length = 150 # Number of historical points to show for object trails

# Calibration Data
calibration_data_dir = "calibration_data" # Relative to the main script

nn_model_slug = "luxonis/yolov6-nano:r2-coco-512x288" 

label_map = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

confidence_threshold = 0.5

# StereoDepth configuration
stereo_depth_preset = "DEFAULT" # or "HIGH_ACCURACY"
stereo_depth_align_to = "CAM_A" # Align to the RGB camera socket

nn_input_size = (512, 288) # Should match the chosen YOLOv6 model's input size

# HTTP Port for dai.RemoteConnection visualizer
HTTP_PORT = 8082
