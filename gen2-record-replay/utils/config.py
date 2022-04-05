from pathlib import Path
import json

class ConfigParser():

    def __init__(self, config_path):
        self._parse_config(config_path)
        

    def _parse_config(self, config_path):
        # parse config
        config_path = Path(config_path)
        if not config_path.exists():
            raise ValueError("Path {} does not exist!".format(config_path))

        with config_path.open() as f:
            config = json.load(f)
        nn_config = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nn_config:
            self.W, self.H = tuple(map(int, nn_config.get("input_size").split('x')))

        # extract metadata
        metadata = nn_config.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchor_masks = metadata.get("anchor_masks", {})
        self.iou_threshold = metadata.get("iou_threshold", {})
        self.confidence_threshold = metadata.get("confidence_threshold", {})

        nn_mappings = config.get("mappings", {})
        self.labels = nn_mappings.get("labels", {})
