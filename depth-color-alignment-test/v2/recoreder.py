from camera import Camera
import cv2
import depthai as dai
import numpy as np
from pathlib import Path

found, device_info = dai.Device.getAnyAvailableDevice()

if not found:
    print("No device found")
    exit(1)

camera = Camera(device_info, 0, show_video=False)

i = 0

recordings_path = Path(f"./recordings/{camera.mxid}")
recordings_path.mkdir(parents=True, exist_ok=True)

while True:
    key = cv2.waitKey(1)

    camera.update()

    if camera.image_frame is not None and camera.depth_visualization_frame is not None:
        cv2.imshow("Image", camera.image_frame)
        cv2.imshow("Depth", camera.depth_visualization_frame)

        if i > 0:
            print(f"Recording {i}...")
            p = recordings_path / str(i)
            p.mkdir(parents=True, exist_ok=True)
            np.save(p / f"image.npy", camera.image_frame)
            np.save(p / f"depth.npy", camera.depth_frame)
            i -= 1

    if key == ord('r'):
        i = 10
    

    if key == ord('q'):
        break
