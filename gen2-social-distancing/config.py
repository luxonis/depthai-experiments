import logging
import os
import sys
from concurrent_log_handler import ConcurrentRotatingFileHandler

root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logfile = os.path.abspath("camera.log")
# Rotate log after reaching 512K, keep 5 old copies.
rotateHandler = ConcurrentRotatingFileHandler(logfile, "a", 512*1024, 5)
rotateHandler.setFormatter(formatter)
root.addHandler(rotateHandler)
root.setLevel(logging.INFO)
root.info("Logging system initialized, kept in file {}...".format(logfile))

MODEL_NAME = "person-detection-retail-0013"
DEBUG = os.getenv('DEBUG', 'true') not in ('false', '0')
