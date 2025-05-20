import numpy as np
import cv2

def preprocess_image(image, target_shape):
    target_h, target_w = target_shape
    h, w = image.shape[:2]

    # Resize to fit within target
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Ensure 3 channels
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    elif resized.shape[2] == 1:
        resized = np.repeat(resized, 3, axis=2)

    # Pad to target
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # NCHW output
    return canvas.transpose(2, 0, 1)[None, ...]


def display_disparity(disparity, window_name, scale=255.0):
    normalized = cv2.normalize(disparity, None, 0, scale, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imshow(window_name, colored)