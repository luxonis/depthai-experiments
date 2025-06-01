bev_width = 800  # Width of the Bird's-Eye View window in pixels
bev_height = 800 # Height of the Bird's-Eye View window in pixels
bev_scale = 40   # Pixels per meter for the Bird's-Eye View (e.g., 1 meter = 40 pixels)
trail_length = 150 # Number of hstorical points to show for object trails

# Calibration Data
calibration_data_dir = "calibration_data" 

nn_model_slug = "luxonis/yolov6-nano:r2-coco-512x288" 

nn_input_size = (512, 288) # Should match the chosen YOLOv6 model's input size

HTTP_PORT = 8082
