import numpy as np
import cv2
from enum import Enum
from typing import Tuple, Optional

class RectFormat(Enum):
    XYWH = 0   # [x, y, w, h]
    XYXY = 1   # [(x1, y1), (x2, y2)]

Color = Tuple[int, int, int]  # only 3‐tuples of ints

def _validate_color(color: Color) -> None:
    if not (
        isinstance(color, tuple) and
        len(color) == 3 and
        all(isinstance(c, int) for c in color)
    ):
        raise ValueError(f"color must be a 3‐tuple of ints, got {color!r}")

def draw_rect(
    img: np.ndarray,
    rect: Optional[np.ndarray],
    *,
    rect_format: RectFormat = RectFormat.XYWH,
    color: Color = (0, 255, 0),
    label: Optional[str] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    # runtime check
    _validate_color(color)

    # no-op if rect is None or all zeros
    if rect is None or not np.any(rect):
        return img

    # color is now guaranteed to be 3 ints in [0,255]
    col = tuple(color)

    # compute corners
    if rect_format is RectFormat.XYWH:
        x, y, w, h = rect
        pt1, pt2 = (int(x), int(y)), (int(x+w), int(y+h))
    else:
        (x1, y1), (x2, y2) = rect
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))

    # draw rectangle
    cv2.rectangle(img, pt1, pt2, col, thickness)

    # draw label if given
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), base = cv2.getTextSize(label, font, font_scale, font_thickness)
        y0 = max(pt1[1], text_h + base)
        # label background
        cv2.rectangle(
            img,
            (pt1[0], y0 - text_h - base),
            (pt1[0] + text_w, y0 + base//2),
            col,
            thickness=-1
        )
        # label text in white
        cv2.putText(
            img, label, (pt1[0], y0 - base//2),
            font, font_scale, (255,255,255),
            font_thickness, lineType=cv2.LINE_AA
        )

    return img

def generate_overlay(frame, mask, weight=0.5, channel = 0):

    purple_overlay = np.zeros_like(frame, dtype=np.uint8) 
    purple_overlay[np.where(mask)] = [255, 0, 255]  
    frame = cv2.addWeighted(frame, 0.7, purple_overlay, 0.3, 0)

    return frame

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def resize_and_pad(img, target_size, pad_color=0):
    # Compute the aspect ratio of the original and target image
    original_aspect = img.shape[1] / img.shape[0]
    target_aspect = target_size[0] / target_size[1]

    if original_aspect > target_aspect:
        # Original has a wider aspect than target. Resize based on width.
        new_w = target_size[0]
        new_h = int(new_w / original_aspect)
    else:
        # Original has a taller aspect or equal to target. Resize based on height.
        new_h = target_size[1]
        new_w = int(new_h * original_aspect)

    # Resize the image
    img_resized = cv2.resize(img, (new_w, new_h))

    # Compute the padding required
    pad_vert = target_size[1] - new_h
    pad_horz = target_size[0] - new_w

    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horz // 2
    pad_right = pad_horz - pad_left

    # Pad the image
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)

    return img_padded

def reverse_resize_and_pad(padded_img, original_size, modified_size, pad_color=0):
    original_width, original_height = original_size
    modified_width, modified_height = modified_size
    
    # Compute the aspect ratio of the original and target image
    original_aspect = original_width / original_height
    modified_aspect = modified_width / modified_height

    if original_aspect > modified_aspect:
        # Original has a wider aspect than target. Resize based on width.
        new_w = modified_width
        new_h = int(new_w / original_aspect)
    else:
        # Original has a taller aspect or equal to target. Resize based on height.
        new_h = modified_height
        new_w = int(new_h * original_aspect)

    # Compute padding
    pad_vert = modified_height - new_h
    pad_horz = modified_width - new_w

    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horz // 2
    pad_right = pad_horz - pad_left

    # Remove padding by cropping
    cropped_img = padded_img[pad_top:modified_height - pad_bottom, pad_left:modified_width - pad_right]

    # Resize back to original dimensions
    # original_img = cv2.resize(cropped_img, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_img = cv2.resize(cropped_img, (original_width, original_height))
    
    return original_img

def center_crop(img, target_size):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    crop_img = img[center_y - target_size[1]//2:center_y + target_size[1]//2, center_x - target_size[0]//2:center_x + target_size[0]//2]
    return crop_img


def transform_bbox(bbox, original_shape, input_shape):
    """
    Transforms a single bounding box from neural network input shape to the original image space.

    Parameters:
        bbox (list or np.ndarray): Bounding box [x_center, y_center, width, height].
        original_shape (tuple): Original image dimensions (width, height).
        input_shape (tuple): Neural network input shape (width, height).

    Returns:
        np.ndarray: Transformed bounding box [xmin, ymin, xmax, ymax] in the original image space.
    """
    orig_width, orig_height = original_shape
    input_width, input_height = input_shape

    # Compute scaling and padding factors
    scale = min(input_width / orig_width, input_height / orig_height)
    pad_x = (input_width - scale * orig_width) / 2
    pad_y = (input_height - scale * orig_height) / 2

    # Scale bbox to input shape
    bbox = np.array(bbox, dtype=float)
    bbox[0::2] *= input_width  # Scale x-coordinates
    bbox[1::2] *= input_height  # Scale y-coordinates

    # Convert center-based bbox to corner-based bbox
    xmin = bbox[0] - bbox[2] / 2
    ymin = bbox[1] - bbox[3] / 2
    xmax = bbox[0] + bbox[2] / 2
    ymax = bbox[1] + bbox[3] / 2

    # Reverse padding and scaling to original image space
    xmin = (xmin - pad_x) / scale
    ymin = (ymin - pad_y) / scale
    xmax = (xmax - pad_x) / scale
    ymax = (ymax - pad_y) / scale

    # Clip coordinates to original image bounds
    xmin = max(0, min(orig_width, xmin))
    ymin = max(0, min(orig_height, ymin))
    xmax = max(0, min(orig_width, xmax))
    ymax = max(0, min(orig_height, ymax))

    return np.array([xmin, ymin, xmax, ymax], dtype=int)

# try for optimization

def compute_resize_params(original_size, modified_size):
    original_width, original_height = original_size
    modified_width, modified_height = modified_size

    # Compute the aspect ratio of the original and target image
    original_aspect = original_width / original_height
    modified_aspect = modified_width / modified_height

    if original_aspect > modified_aspect:
        # Original has a wider aspect than target. Resize based on width.
        new_w = modified_width
        new_h = int(new_w / original_aspect)
    else:
        # Original has a taller aspect or equal to target. Resize based on height.
        new_h = modified_height
        new_w = int(new_h * original_aspect)

    # Compute padding
    pad_vert = modified_height - new_h
    pad_horz = modified_width - new_w

    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horz // 2
    pad_right = pad_horz - pad_left

    return pad_top, pad_bottom, pad_left, pad_right

def reverse_resize_and_pad2(padded_img, resize_params, original_size, modified_size, pad_color=0):

    pad_top, pad_bottom, pad_left, pad_right = resize_params
    original_width, original_height = original_size
    modified_width, modified_height = modified_size

    # Remove padding by cropping
    print(f"Image shape: {padded_img.shape}, Modified size: ({modified_width}, {modified_height}), Resize Params: {resize_params}")

    cropped_img = padded_img[pad_top:modified_height - pad_bottom, pad_left:modified_width - pad_right]

    # Resize back to original dimensions
    original_img = cv2.resize(cropped_img, (original_width, original_height))
    
    return original_img
