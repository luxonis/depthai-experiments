import base64
import io
import time
from typing import List

import cv2
import numpy as np
import requests
from PIL import Image

from .annotations import make_voc_annotations


class RoboflowUploader:
    # Thin wrapper over Roboflow's upload API

    def __init__(self, dataset_name: str, api_key: str):

        self._dataset_name = dataset_name
        self._api_key = api_key

    def upload_image(self, arr: np.ndarray, fname: str):
        # Uploads an `arr`, returns Roboflow's image id

        # BGR -> RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        # Load Image with PIL
        image = Image.fromarray(arr)

        # JPEG encoding
        buf = io.BytesIO()
        image.save(buf, quality=90, format="JPEG")

        # Base 64 Encode
        img_str = base64.b64encode(buf.getvalue())
        img_str = img_str.decode("ascii")

        # Construct the URL
        upload_url = "".join(
            [
                f"https://api.roboflow.com/dataset/{self._dataset_name}/upload",
                f"?api_key={self._api_key}",
                f"&name={fname}.jpg",  # For example 1640677054993.jpg
                "&split=train",
            ]
        )

        # POST to the API
        r = requests.post(
            upload_url,
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        # Get image id from response
        img_id = r.json().get("id")

        # Print debug info
        if r.status_code == 200:
            print(f"INFO: Uploaded image with ID: {img_id}")
        else:
            print(f"ERROR: Failed to upload image. HTTP status: {r.status_code}")
            print(r.json())

        return img_id

    def upload_annotation(
        self, image_id, fname, labels: List[str], bboxes: List[List[int]], img_w=300, img_h=300
    ):
        # Uploads a VOC annotation string for given `image_id`
        # The annotation will be stored under `fname.xml`
        # Returns `True` if upload succeeded, `False` otherwise

        annotation_str = make_voc_annotations(labels, bboxes, img_w, img_h)

        upload_url = "".join(
            [
                f"https://api.roboflow.com/dataset/{self._dataset_name}/annotate/{image_id}",
                f"?api_key={self._api_key}",
                f"&name={fname}.xml",
            ]
        )

        # POST to the API
        r = requests.post(
            upload_url, data=annotation_str, headers={"Content-Type": "text/plain"}
        )

        if r.status_code == 200:
            print(f"INFO: Uploaded annotation for image ID: {image_id}")
        else:
            print(f"ERROR: failed annotation for image ID: {image_id}")
            print(r.json())


if __name__ == "__main__":

    # Testing code:

    arr = (np.random.random((500, 500, 3)) * 255).astype(np.uint8)
    unique_id = int(1000 * time.time())

    uploader = RoboflowUploader(
        dataset_name="oak-dataset2", api_key="vkIkZac3CXvp0RZ31B3f"
    )

    start = time.perf_counter()
    img_id = uploader.upload_image(arr, unique_id)

    uploader.upload_annotation(
        img_id,
        unique_id,
        ["helmet", "helmet"],
        [[179, 85, 231, 144], [112, 145, 135, 175]],
    )

    print(time.perf_counter() - start)
