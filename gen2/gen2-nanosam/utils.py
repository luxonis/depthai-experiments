import numpy as np
import cv2

def generate_overlay(frame, mask, weight=0.5):
    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
    frame = (
        frame * (1 - mask) + (frame * weight + np.array([255, 0, 255]) * weight) * mask
    ).astype(np.uint8)

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

def center_crop(img, target_size):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    crop_img = img[center_y - target_size[1]//2:center_y + target_size[1]//2, center_x - target_size[0]//2:center_x + target_size[0]//2]
    return crop_img