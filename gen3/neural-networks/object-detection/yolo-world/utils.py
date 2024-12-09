import os
import cv2
import numpy as np
import requests


def draw_detections(frame, detections, class_names):
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    for detection in detections:
        if detection.label > len(class_names) - 1:
            continue

        bbox = frame_norm(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )
        x1, y1, x2, y2 = bbox

        color = (
            int(detection.label * 73 % 255),
            int(detection.label * 157 % 255),
            int(detection.label * 241 % 255),
        )

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        alpha = 0.4  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = (
            f"{class_names[detection.label]}: {int(detection.confidence * 100)}%"
        )
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = x1, y1 - 10

        cv2.rectangle(
            frame,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return frame


def download_model(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading model from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Model saved to {save_path}.")
        else:
            raise Exception(
                f"Failed to download model. Status code: {response.status_code}"
            )
    else:
        print(f"Model already exists at {save_path}.")

    return save_path
