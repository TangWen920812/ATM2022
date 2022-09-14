from __future__ import absolute_import

from .common import logger
from .lung_segmentor import getLungMask, Status

__version__ = "3.0.0"
logger.info("lung_segmentor_itk version %s initiated." % __version__)
