import os
import logging
from pathlib import Path

import clip
import cv2
import numpy as np
import pkg_resources
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx : min(ndx + bs, l)]


class Clip_Embedder(object):
    """
    Clip_Embedder loads a CLIP model of specified architecture, outputting a feature of size 1024.

    Params
    ------
    - model_name (optional, str) : CLIP model to use
    - model_wts_path (optional, str): Optional specification of path to CLIP model weights. Defaults to None and look for weights in `deep_sort_realtime/embedder/weights` or clip will download from internet into their own cache.
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    - gpu (optional, Bool) : boolean flag indicating if gpu is enabled or not, defaults to True
    """

    def __init__(
        self,
        model_name="ViT-B/32",
        model_wts_path=None,
        max_batch_size=16,
        bgr=True,
        gpu=True,
    ):
        if model_wts_path is None:
            assert model_name in clip.available_models()

            weights_name = model_name.replace("/", "-")
            weights_path = (
                Path(__file__).parent.resolve() / "weights" / f"{weights_name}.pt"
            )
            if weights_path.is_file():
                model_wts_path = str(weights_path)
            else:
                model_wts_path = model_name

        self.device = "cuda" if gpu else "cpu"
        self.model, self.img_preprocess = clip.load(model_wts_path, device=self.device)
        self.model.eval()

        self.max_batch_size = max_batch_size
        self.bgr = bgr

        logger.info("Clip Embedder for Deep Sort initialised")
        logger.info(f"- gpu enabled: {gpu}")
        logger.info(f"- max batch size: {self.max_batch_size}")
        logger.info(f"- expects BGR: {self.bgr}")
        logger.info(f"- model name: {model_name}")

        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 1024)

        """
        if not np_images:
            return []

        if self.bgr:
            np_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in np_images]

        pil_images = [
            self.img_preprocess(Image.fromarray(rgb)).to(self.device)
            for rgb in np_images
        ]

        all_feats = []
        for this_batch in _batch(pil_images, bs=self.max_batch_size):
            batch = torch.stack(this_batch, 0)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
            all_feats.extend(feats.cpu().data.numpy())
        return all_feats
