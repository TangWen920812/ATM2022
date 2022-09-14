import logging
import os

_LUNG_INTERSITY_THRESHOLD = float(os.getenv("LUNG_INTERSITY_THRESHOLD", "-320"))
logger = logging.getLogger("lung_segmentor_itk")
logger.addHandler(logging.NullHandler())
